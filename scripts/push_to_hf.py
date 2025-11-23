#push the model weights to hugging face

from huggingface_hub import HfApi, login
import argparse


def push_to_hf(args):
    token = args.token
    login(token)
    folder_path = args.folder_path
    repo_id = args.repo_id
    repo_type = args.repo_type
    api = HfApi()

    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, required=True)
    parser.add_argument('--repo_id', type=str, required=True)
    parser.add_argument('--token', type=str, required=True) 
    parser.add_argument('--repo_type', type=str, default='datasets')
    args = parser.parse_args()
    push_to_hf(args)

if __name__ == "__main__":
    main()
    