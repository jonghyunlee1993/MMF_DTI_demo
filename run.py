import gradio as gr
import torch.nn.functional as F

from models.model import DTI
from utils.preprocessing import *

def load_tokenizers():
    mol_tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    prot_tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert_bfd", do_lower_case=False)
    text_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    return mol_tokenizer, prot_tokenizer, text_tokenizer
    
def load_models(prot_tokenizer):
    
    config = BertConfig(
        vocab_size=prot_tokenizer.vocab_size,
        hidden_size=1024,
        num_hidden_layers=2,
        num_attention_heads=16,
        intermediate_size=4096,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=545 + 2,
        type_vocab_size=1,
        pad_token_id=0,
        position_embedding_type="absolute"
    )

    prot_student_encoder = BertModel(config)
    prot_teacher_encoder = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")
    mol_encoder = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    text_encoder = AutoModel.from_pretrained("jonghyunlee/UniProtBERT")
    
    dti_model = DTI(mol_encoder, prot_student_encoder, text_encoder)
    dti_model.load_state_dict(torch.load("weights/MMF_DTI.pt", map_location=torch.device('cpu')), strict=False)

    return dti_model.eval(), prot_teacher_encoder.eval()

def drug_target_interaction(SMILES, FASTA, function_description, use_gpt, api_key):
    mol_tokenizer, prot_tokenizer, text_tokenizer = load_tokenizers()
    dti_model, prot_teacher_model = load_models(prot_tokenizer)
    
    function_description = summarize_text(function_description, use_gpt, api_key)
    
    mol_property = compute_mol_property(SMILES)
    
    tokenized_smiles, tokenized_fasta, tokenized_text = tokenize_inputs(
        SMILES, FASTA, function_description, 
        mol_tokenizer, prot_tokenizer, text_tokenizer,
        use_gpt, api_key
    )
    
    prot_feat_teacher = get_prot_feat(tokenized_fasta, prot_teacher_model)

    dti_model.eval()
    with torch.no_grad():
        output, _ = dti_model(tokenized_smiles, tokenized_fasta, prot_feat_teacher, tokenized_text, mol_property)
        output = F.sigmoid(output)

    return round(output[0].numpy().tolist(), 4)

def gradio_app():
    with gr.Blocks() as demo:
        gr.Markdown("Drug-Target Interaction Prediction")

        with gr.Row():
            with gr.Column():
                SMILES = gr.Textbox(label="Drug Sequence (SMILES)")
                FASTA = gr.Textbox(label="Target Sequence (FASTA)", lines=3)
                functional_description = gr.Textbox(label="Target Function Description", lines=3)
                use_gpt = gr.Radio(label="Use GPT for summarization", choices=["Yes", "No"], value="No")
                api_key = gr.Textbox(label="GPT API Key", type="password")

                btn = gr.Button("Predict")
            
            with gr.Column():
                output = gr.Textbox(label="Prediction", lines=1)

        btn.click(fn=drug_target_interaction, inputs=[SMILES, FASTA, functional_description, use_gpt, api_key], outputs=output)

    demo.launch(debug=True)

if __name__ == "__main__":
    gradio_app()