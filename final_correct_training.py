import json
import os
import shutil
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset

def load_existing_dataset():
    """Load the existing improved short dataset"""
    print("📝 Loading existing improved short dataset...")
    
    qa_pairs = []
    with open("qa_dataset_improved_short.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            qa_pairs.append(json.loads(line.strip()))
    
    print(f"✅ Loaded {len(qa_pairs)} Q&A pairs from qa_dataset_improved_short.jsonl")
    
    # Show the exact answers we want to learn
    print("\n📋 EXACT ANSWERS TO LEARN:")
    for i, qa in enumerate(qa_pairs):
        print(f"Q{i+1}: {qa['question']}")
        print(f"A{i+1}: {qa['answer']}")
        print(f"Length: {len(qa['answer'])} characters")
        print("-" * 50)
    
    return qa_pairs

def train_with_exact_answers():
    """Train the model to learn the exact answers from the dataset"""
    print("\n🔧 TRAINING WITH EXACT ANSWERS")
    print("=" * 50)
    
    # Remove old model completely
    if os.path.exists("fine_tuned_model"):
        print("🗑️ Removing old fine-tuned model...")
        shutil.rmtree("fine_tuned_model")
    
    # Load the exact dataset
    qa_pairs = load_existing_dataset()
    
    # Create training data with simple, direct mapping
    training_data = []
    for qa in qa_pairs:
        # Create multiple training examples for each Q&A pair
        # Format 1: Direct question -> exact answer
        training_data.append({
            "input_text": qa['question'],
            "target_text": qa['answer']
        })
        
        # Format 2: Question with prefix -> exact answer
        training_data.append({
            "input_text": f"Question: {qa['question']}",
            "target_text": qa['answer']
        })
        
        # Format 3: Question with context -> exact answer
        training_data.append({
            "input_text": f"Context: {qa['context']}\nQuestion: {qa['question']}",
            "target_text": qa['answer']
        })
        
        # Format 4: Instruction format -> exact answer
        training_data.append({
            "input_text": f"Answer this question: {qa['question']}",
            "target_text": qa['answer']
        })
        
        # Format 5: Simple format -> exact answer (repeat for emphasis)
        training_data.append({
            "input_text": qa['question'],
            "target_text": qa['answer']
        })
    
    print(f"✅ Created {len(training_data)} training examples")
    
    # Create dataset
    dataset = Dataset.from_list(training_data)
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    print(f"✅ Dataset: {len(dataset['train'])} train, {len(dataset['test'])} test")
    
    # Load base model
    print("📥 Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained("base_model")
    model = AutoModelForSeq2SeqLM.from_pretrained("base_model")
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Simple tokenization function
    def tokenize_function(examples):
        # Tokenize inputs
        inputs = tokenizer(
            examples["input_text"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize targets
        targets = tokenizer(
            examples["target_text"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        inputs["labels"] = targets["input_ids"]
        return inputs
    
    # Tokenize datasets
    print("🔤 Tokenizing datasets...")
    tokenized_train = dataset["train"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    tokenized_test = dataset["test"].map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["test"].column_names
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments optimized for exact learning
    training_args = TrainingArguments(
        output_dir="fine_tuned_model",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=10,
        learning_rate=3e-5,  # Moderate learning rate
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=50,  # More epochs for exact learning
        fp16=False,  # Disable mixed precision
        dataloader_pin_memory=False,
        logging_steps=1,
        warmup_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        prediction_loss_only=True,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("🚀 Starting training with exact answers...")
    trainer.train()
    
    # Save
    print("💾 Saving model...")
    trainer.save_model()
    tokenizer.save_pretrained("fine_tuned_model")
    
    print("✅ Training completed!")
    return "fine_tuned_model"

def test_exact_answers():
    """Test the model against the exact expected answers"""
    print("\n🧪 TESTING EXACT ANSWERS")
    print("=" * 50)
    
    # Load the expected answers
    expected_answers = {}
    with open("qa_dataset_improved_short.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            qa = json.loads(line.strip())
            expected_answers[qa['question']] = qa['answer']
    
    test_questions = [
        "What is science?",
        "What are the main branches of science?",
        "What is physics?",
        "What is chemistry?",
        "What is the scientific method?"
    ]
    
    for model_name, model_path in [("Base Model", "base_model"), ("Fine-tuned Model (Exact)", "fine_tuned_model")]:
        print(f"\n📋 {model_name.upper()}:")
        print("-" * 50)
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            
            for question in test_questions:
                expected = expected_answers.get(question, "No expected answer")
                
                # Try different input formats
                input_formats = [
                    question,
                    f"Question: {question}",
                    f"Answer this question: {question}"
                ]
                
                best_answer = ""
                for input_text in input_formats:
                    inputs = tokenizer(
                        input_text, 
                        return_tensors="pt", 
                        max_length=128, 
                        truncation=True, 
                        padding=True
                    )
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                    
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_length=128,
                            num_beams=3,
                            early_stopping=True,
                            do_sample=False,
                            pad_token_id=tokenizer.pad_token_id
                        )
                    
                    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    answer = answer.strip()
                    
                    if len(answer) > len(best_answer):
                        best_answer = answer
                
                print(f"Q: {question}")
                print(f"Expected: {expected}")
                print(f"Generated: {best_answer}")
                print(f"Match: {'✅' if best_answer.strip() == expected.strip() else '❌'}")
                print(f"Length: {len(best_answer)} characters")
                print()
                
        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")

def main():
    print("🔧 FINAL CORRECT TRAINING")
    print("This will train the model to give the exact answers from qa_dataset_improved_short.jsonl")
    print("=" * 50)
    
    # Train with exact answers
    train_with_exact_answers()
    
    # Test against expected answers
    test_exact_answers()
    
    print("\n🎉 FINAL TRAINING COMPLETED!")
    print("The fine-tuned model should now give the exact answers from the dataset")
    print("You can now run: streamlit run app.py")

if __name__ == "__main__":
    main() 