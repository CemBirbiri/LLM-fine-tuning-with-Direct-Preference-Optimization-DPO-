# LLM fine-tuning with Direct Preference Optimization (DPO) with code

![Screenshot 2025-03-01 145638](https://github.com/user-attachments/assets/fdc59b58-914f-446f-bf68-b26c4ea7b098)

The image is created using GPT 4o

In this article we will cover how to fine-tune LLMs with Direct Preference Optimization (DPO) with codes. 

### What is DPO?

Training language models to align with human preferences is crucial for improving their usefulness and safety. Traditionally, this has been achieved through Reinforcement Learning with Human Feedback (RLHF) using Proximal Policy Optimization (PPO). While effective, this approach is computationally demanding and complex. Direct Preference Optimization (DPO) offers a more efficient alternative by directly aligning a model's behavior with human preferences without relying on explicit reward modeling or reinforcement learning techniques.

### How DPO Works

`1. Collecting Human Preferences:` DPO begins by gathering human feedback on pairs of model responses, where one response is preferred over the other, referred as CHOSEN or REJECTED responses. These collected data is based on human feedback.

   
`2. Implicit Reward Model:` DPO does not explicitly use a reward function like traditional Reinforcement Learning with Human Feedback (RLHF) approaches such as Proximal Policy Optimization (PPO). Instead, DPO directly optimizes a policy using pairwise preference data without requiring a reward model.

`3. Policy Adjustment:` The model is updated to increase the likelihood of generating preferred responses while reducing the probability of less favorable ones. This adjustment is done dynamically for each example, ensuring that the model improves without introducing instability.

`4. Efficient Optimization:` Unlike traditional reinforcement learning, DPO directly integrates preference-based loss functions into standard training techniques. This simplifies the process, making it more scalable and computationally efficient.


By eliminating the need for explicit reward modeling and complex reinforcement learning algorithms, DPO streamlines preference-based fine-tuning, making it a practical solution for aligning AI models with human expectations.

![xxx](https://github.com/user-attachments/assets/9340e549-11f8-4aa1-ad57-ec88e700b6a3)

The image is taken [from the original DPO paper.](https://arxiv.org/pdf/2305.18290)

**Objectives**

- Grasp the fundamentals of Direct Preference Optimization (DPO) and distinguish it from Proximal Policy Optimization (PPO).
- Construct a dataset tailored for DPO.
- Understand the required data format for DPO applications.
- Implement DPO by following a structured, step-by-step approach using the trl library.
- Define training parameters, initialize a quantized LoRA base model, and train it using a DPO trainer.
- Evaluate the LLM's performance before and after applying DPO.

Install libraries

```python
!pip install torch
!pip install trl # for optimization training
!pip install peft # for creating LoRA architecture
!pip install matplotlib
```

Import libraries

```python
import multiprocessing
import os
import requests
import tarfile
import pandas as pd
import matplotlib.pyplot as plt

import torch
from datasets import load_dataset

from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer,TrainingArguments, GPT2Tokenizer, set_seed, GenerationConfig
from trl import DPOConfig, DPOTrainer
```

Create and configure the model and tokenizer

```python
We will use the gpt2 as our model.
# Load model
model = AutoModelForCausalLM.from_pretrained("gpt2")

# Load the reference model.
model_ref = AutoModelForCausalLM.from_pretrained("gpt2")

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define the end of sequence token
tokenizer.pad_token = tokenizer.eos_token
# Define the padding side as right to handle the overflow problem with FP16 training
tokenizer.padding_side = "right"

# Disable the use of the cache during the model's forward pass
model.config.use_cache = False
```

### Dataset

We will use the `[argilla/ultrafeedback-binarized-preferences-cleaned](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned)` dataset from HuggingFace. This dataset contains prompts and selected feedbacks labeled as CHOSEN or REJECTED. This dataset will serve as the human feedback.

```python
# Load the dataset 
dataset = load_dataset("argilla/ultrafeedback-binarized-preferences-cleaned")
```

Process the data, split as train-test, choose only CHOSEN and REJECTED parts.

```python
import multiprocessing
from datasets import load_dataset

# Reduce dataset volume by selecting the first 50 examples from each split
for key in dataset.keys():
    dataset[key] = dataset[key].select(range(50))

# Function to process data
def process(data_point):
    # Remove unwanted columns safely
    for col in ['source', 'chosen-rating', 'chosen-model', 'rejected-rating', 'rejected-model']:
        data_point.pop(col, None)  # Avoids errors if the column is missing

    # Extract the last response content
    data_point["chosen"] = data_point["chosen"][-1]["content"]
    data_point["rejected"] = data_point["rejected"][-1]["content"]

    return data_point

# Apply mapping to process al the rows
dataset = dataset.map(process, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False,)

# Split train dataset: 90% for training, 10% for testing
ds_train_test = dataset["train"].train_test_split(test_size=0.1)

# Assign new splits
dataset["train"], dataset["test"] = ds_train_test["train"], ds_train_test["test"]

train_dataset = dataset["train"]
eval_dataset = dataset["test"]
```
```python
train_dataset[0]
```

```python
Output:

{'prompt': "Let's play a puzzle game! Can you connect the dots and figure out how rising global temperatures, ...",
 'chosen': "Of course, I'd be happy to help you solve the puzzle and understand...",
 'rejected': 'Indeed, the interlinked impacts of climate change on the environment, including ...'}
```

Next, define LoRAConfig for efficient fine-tuning.

```python
from peft import LoraConfig

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
peft_config = LoraConfig(
        # Rank parameter defining the size of the LoRA update matrices
        r=4,
        # Layers where LoRA is applied (projection and attention layers)
        target_modules=['c_proj', 'c_attn'],
        # Specifies the task type (Causal Language Modeling)
        task_type="CAUSAL_LM",
        # Scaling factor that controls the impact of LoRA weights
        lora_alpha=8,
        # Dropout applied to LoRA layers to prevent overfitting
        lora_dropout=0.1,
        # Bias configuration (no additional bias is applied)
        bias="none",
)
```python

**DPO configuration**

We will use the DPOConfig to arrange the training parameters.

```python
from trl import DPOConfig

# Configure Direct Preference Optimization (DPO) training parameters
training_args = DPOConfig(
    # Temperature parameter controlling the sharpness of preference modeling in DPO loss
    # Typically set in the range of 0.1 to 0.5 for stable optimization
    beta=0.1,

    # Directory to save model checkpoints and outputs
    output_dir="dpo",

    # Number of times the entire dataset is passed through during training
    num_train_epochs=1,

    # Batch size for training per device (smaller values prevent memory issues)
    per_device_train_batch_size=1,

    # Batch size for evaluation per device
    per_device_eval_batch_size=1,

    # Ensures dataset columns not required by the model are not removed
    remove_unused_columns=False,

    # Log training progress every `logging_steps` steps
    logging_steps=10,

    # Number of steps to accumulate gradients before performing an optimizer update
    gradient_accumulation_steps=1,

    # Learning rate for the optimizer (affects model weight updates)
    learning_rate=1e-4,

    # Defines when evaluation occurs (e.g., after each epoch)
    evaluation_strategy="epoch",

    # Number of warmup steps before reaching the full learning rate
    warmup_steps=2,

    # Whether to use 16-bit floating point precision (reduces memory usage)
    fp16=False,

    # Save model checkpoints every `save_steps` steps
    save_steps=500,

    # Backend for tracking training progress ('none' disables logging, alternatives include 'wandb' or 'tensorboard')
    report_to='none',
    max_prompt_length=512,            # Maximum input prompt length
    max_length=512,                    # Maximum sequence length (prompt + response)
)
```

**DPO training**

We will create the training object DPOTrainer. When using LoRA (Low-Rank Adaptation) in DPOTrainer for fine-tuning a base model, we set the model_ref parameter to None for the following reasons:

`LoRA Only Updates the Adapters, Not the Base Model:` LoRA fine-tunes a pretrained base model by introducing low-rank trainable adapters, while keeping the base model frozen. The model_ref in DPO (Direct Preference Optimization) is typically used for comparison between the fine-tuned and base model outputs. Since the base model itself remains unchanged, there's no need for a reference model.

`Avoiding Redundant Memory Usage:` If model_ref is not set to None, DPOTrainer will load two versions of the model: The fine-tuned model (with LoRA adapters applied) A reference model (baseline for reward comparison) This doubles the memory usage unnecessarily, especially when LoRA only modifies a small subset of parameters.

`DPO Can Use the Frozen Base Model Internally:` By setting model_ref=None, DPOTrainer can still compare outputs using the original frozen base model (without explicitly loading a second instance). This is useful in scenarios where the base model itself acts as the reference.

```python

# Set the padding token to be the same as the EOS (end-of-sequence) token
tokenizer.pad_token = tokenizer.eos_token

# Initialize the Direct Preference Optimization (DPO) Trainer
trainer = DPOTrainer(
        model=model,                    # The model to be fine-tuned
        ref_model=None,                  # No reference model needed with LoRA
        args=training_args,               # DPOConfig containing all hyperparameters, including beta
        train_dataset=train_dataset,      # Dataset for training
        eval_dataset=eval_dataset,        # Dataset for evaluation
        tokenizer=tokenizer,              # Tokenizer to process input/output text
        peft_config=peft_config,          # LoRA configuration for efficient fine-tuning      
)
```

**Train the model**

We will train the model only 1 epoch because this notebook is implemented using Google Colab and using CPU.

```python
# Start the training process
trainer.train()
```

Let's retrieve and plot the training loss versus evaluation loss.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Convert log history to a DataFrame
log = pd.DataFrame(trainer.state.log_history)

# Filter training and evaluation loss entries
train_log = log.dropna(subset=['loss'])
eval_log = log.dropna(subset=['eval_loss'])

# Plot training and evaluation losses
plt.plot(train_log["epoch"], train_log["loss"], label="Train Loss")
plt.plot(eval_log["epoch"], eval_log["eval_loss"], label="Eval Loss")

# Add legend and show plot
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Evaluation Loss")
plt.show()
```

Note that we only trained the DPO model 1 eopch due to Colab environment and insufficient computatinal power.


Evaluate the DPO model

```python
# Load again the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# Set a seed for reproducibility
set_seed(31)

# Define parameters for response genearation
generation_config = GenerationConfig(
        # Use sampling to generate diverse text
        do_sample=True,
        # Top-k sampling parameter
        top_k=1,
        # Temperature parameter to control the randomness of the generated text
        temperature=0.1,
        # Maximum number of new tokens to generate
        max_new_tokens=25,
        # Use the end-of-sequence token as the padding token
        pad_token_id=tokenizer.eos_token_id
    )

# Define the input prompt for text generation
PROMPT = "Is eating animals cruel?"
# Tokenize prompt
inputs = tokenizer(PROMPT, return_tensors='pt')

# Generate response from DPO model
outputs = model.generate(**inputs, generation_config=generation_config)
# Decode the generated text
print("DPO response:",tokenizer.decode(outputs[0], skip_special_tokens=True))

# Load the pre-trained GPT2 
gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
# Generate response
outputs = gpt2_model.generate(**inputs, generation_config=generation_config)
# Decode the generated text
print("GPT2 response:",tokenizer.decode(outputs[0], skip_special_tokens=True))
```

```python
Output:

DPO response: Is eating animals cruel?

The answer is yes.

The Humane Society of the United States (HSUS) has been working with the
GPT2 response: Is eating animals cruel?

The answer is yes.

The Humane Society of the United States (HSUS) has been working with the
```

We can see the answers are not perfect and the reason is that we trained the DPO model for only 1 epoch. This notebook is a demonstration of how to implement DPO model and fine-tune your LLM.
