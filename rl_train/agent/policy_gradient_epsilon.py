import torch
from torch.optim import AdamW
import random
from torch.distributions import Categorical
import torch.nn.functional as F

def fgsm_attack(image, epsilon, gradient):
    """
    Fast Gradient Sign Method (FGSM) attack.
    """
    sign_grad = gradient.sign()
    perturbed_image = image + epsilon * sign_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

class PolicyAdvEpsilonAgent:
    def __init__(self, model, lr=1e-4, device=None, epsilon=0.1):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.model = model.to(device)
        self.optimizer = AdamW(model.parameters(), lr=lr)
        self.epsilon = epsilon # Exploration epsilon

    def _get_probs_and_actions(self, states, labels):
        """
        内部辅助函数：计算策略输出和动作
        """
        # 注意：这里不需要检查 self.model.training，因为它只负责前向计算和采样
        policy_output = self.model(states)
        
        temperature = 1.0
        softmax_input = policy_output / temperature
        softmax_input = softmax_input - softmax_input.max(dim=1, keepdim=True)[0]
        
        eps_prob = 1e-6
        action_probs = F.softmax(softmax_input, dim=-1) + eps_prob
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        
        if torch.isnan(action_probs).any():
            action_probs = torch.ones_like(action_probs) / action_probs.size(-1)
        
        # Epsilon-greedy action selection
        if random.random() < self.epsilon and self.model.training: # 训练时才使用探索
            actions = torch.randint(0, action_probs.size(1), (states.size(0),)).to(self.device)
        else:
            dist = Categorical(action_probs)
            actions = dist.sample()
            
        # Log probabilities for PG loss
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze(1) + eps_prob)
        
        return action_probs, log_probs, actions

    def update(self, states, labels) -> tuple[float, float]:
        """
        Standard Policy Gradient update (Clean training).
        """
        self.model.train()
        self.optimizer.zero_grad()
        
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        states = states.to(self.device)
        labels = labels.to(self.device).view(-1)
        
        _, log_probs, actions = self._get_probs_and_actions(states, labels)

        rewards = (actions == labels).float()
        loss = -(log_probs * rewards).mean()
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), 0.0


    def adv_update(self, states, labels, adv_rate, epsilon) -> tuple[float, float]:
        """
        Weighted Double Policy Gradient update (SL-style consistency).
        Adv_rate is the weight (beta) for the adversarial loss L_PG(x_adv).
        """
        
        # --- 1. PREPARE DATA ---
        # 确保 states 具有梯度，用于攻击生成
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32, requires_grad=True)
        elif not states.requires_grad:
            states = states.clone().detach().requires_grad_(True)
            
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        states = states.to(self.device)
        labels = labels.to(self.device).view(-1)

        # 确保模型在训练模式下进行更新
        self.model.train()
        self.optimizer.zero_grad()

        # ----------------------------------------------------------------------
        # --- PART 2: CLEAN LOSS (L_PG(x)) ---
        # ----------------------------------------------------------------------

        # 获取清洁样本的 PG 损失 (L_PG(x))
        _, log_probs_clean, actions_clean = self._get_probs_and_actions(states, labels)
        rewards_clean = (actions_clean == labels).float()
        loss_clean = -(log_probs_clean * rewards_clean).mean()
        
        # 记录清洁损失
        basic_loss_item = loss_clean.item()
        
        # ----------------------------------------------------------------------
        # --- PART 3: GENERATE ADVERSARIAL SAMPLE (Requires L_PG(x) Gradient) ---
        # ----------------------------------------------------------------------
        
        # 1. 计算 L_PG(x) w.r.t input 的梯度
        # 由于我们只关心 input 梯度，这里我们使用 eval() 模式来冻结 BN，避免双重更新。
        original_mode = self.model.training
        self.model.eval()
        
        # IMPORTANT: 只反传到 states，不更新模型权重
        # retain_graph=True 是因为后面需要对 total_loss 再次反传
        loss_clean.backward(retain_graph=True) 
        
        # 2. 生成 Perturbation (使用 L_PG(x) 的梯度)
        perturbed_states = states.detach().clone()
        if epsilon > 0 and states.grad is not None:
            # 应用 FGSM
            perturbed_states = fgsm_attack(states.detach(), epsilon, states.grad.data)
        
        # 恢复模型模式
        if original_mode:
            self.model.train()
        
        # ----------------------------------------------------------------------
        # --- PART 4: ADVERSARIAL LOSS (L_PG(x_adv)) ---
        # ----------------------------------------------------------------------
        
        # 获取对抗样本的 PG 损失 (L_PG(x_adv))
        # 使用 detach() 样本进行前向传播
        _, log_probs_adv, actions_adv = self._get_probs_and_actions(perturbed_states.detach(), labels)
        rewards_adv = (actions_adv == labels).float()
        loss_adv = -(log_probs_adv * rewards_adv).mean()
        
        adv_loss_item = loss_adv.item()
        
        # ----------------------------------------------------------------------
        # --- PART 5: TOTAL LOSS AND OPTIMIZATION (L_Total = L_Clean + Rate * L_Adv) ---
        # ----------------------------------------------------------------------
        
        # 清零所有梯度
        self.optimizer.zero_grad()
        
        # === L_Clean Recalculation ===
        _, log_probs_clean_final, actions_clean_final = self._get_probs_and_actions(states, labels)
        rewards_clean_final = (actions_clean_final == labels).float()
        loss_clean_final = -(log_probs_clean_final * rewards_clean_final).mean()

        # === L_Adv Recalculation ===
        _, log_probs_adv_final, actions_adv_final = self._get_probs_and_actions(perturbed_states.detach(), labels)
        rewards_adv_final = (actions_adv_final == labels).float()
        loss_adv_final = -(log_probs_adv_final * rewards_adv_final).mean()


        # L_Total = L_PG(x) + adv_rate * L_PG(x_adv)
        total_loss = loss_clean_final + adv_rate * loss_adv_final
        
        # 总反传
        total_loss.backward()
        self.optimizer.step()
        
        # 返回 L_clean (Part 2) 和 L_adv (Part 4) 的值用于日志记录
        return basic_loss_item, adv_loss_item

    def select_action(self, state):
        self.model.eval() # 确保 eval mode for inference
        with torch.no_grad():
            state = state.to(self.device)
            action_probs, _, _ = self._get_probs_and_actions(state, None)
            action = torch.argmax(action_probs, dim=-1)
        return action