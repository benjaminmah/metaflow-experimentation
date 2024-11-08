# metaflow-experimentation

## Setup Instructions

1. **Create a Python Virtual Environment**
   ```bash
   python3 -m venv venv
    ```

2. **Activate Virtual Environment**
   ```bash
   source venv/bin/activate
    ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
    ```

3. **Set the Required Environment Variables**
   ```bash
    export WANDB_API_KEY=your_wandb_api_key
    export WANDB_PROJECT=your_wandb_project_name
    export HF_TOKEN=your_hf_token
    ```
    Please ensure your HF token has permissions to the models being used in `template_flow.py`.

5. **Run Command**
   ```bash
   python3 -m template_flow --environment=pypi run
    ```