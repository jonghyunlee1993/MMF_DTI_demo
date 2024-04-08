import re
import torch

from rdkit import Chem
from rdkit.Chem import Descriptors

from openai import OpenAI
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel


def preprocess_text(text):
    # pattern = r'\s*\(?PubMed:\d+(, PubMed:\d+)*\)?\s*'
    pattern_1 = re.compile(r"\s+\(PubMed:\d+(?:, PubMed:\d+)*\)")
    text = re.sub(pattern_1, "", text)

    pattern_2 = re.compile(r"\{ECO:\d+(?:\|\w+:[^,]+(?:,\s*\w+:[^,]+)*)*\}[\.;]")
    text = re.sub(pattern_2, "", text)
    text = text.replace(";", "")

    return text


def compute_mol_property(SMILES):
    try:
        mol = Chem.MolFromSmiles(SMILES)
        descriptor = Descriptors.CalcMolDescriptors(mol)
        property = list(descriptor.values())[:209]
    except:
        property = [0] * 209

    return torch.tensor(property).float()


def summarize_text(text, use_gpt, api_key=None):
    if use_gpt == "Yes" and len(api_key) > 0:
        client = OpenAI(
            api_key=api_key,
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are helpful assistant"},
                {
                    "role": "user",
                    "content": f"Read the given text and shorten it. The maximum length of the summarization should not exceed 512 characters. The output format should follow the output template.\n\nOUTPUT TEMPLATE\n[OUTPUT] [Write the summarized text]\n\nINPUT TEXT: {text}",
                },
            ],
            temperature=0,
            max_tokens=512,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
        output = response.choices[0].message.content
    else:
        output = "[OUTPUT] " + text

    return output


def tokenize_inputs(
    SMILES,
    FASTA,
    function_description,
    mol_tokenizer,
    prot_tokenizer,
    text_tokenizer,
    use_gpt,
    api_key,
):
    tokenized_smiles = mol_tokenizer(
        SMILES, max_length=512, truncation=True, return_tensors="pt"
    )
    tokenized_fasta = prot_tokenizer(
        " ".join(FASTA), max_length=545 + 2, truncation=True, return_tensors="pt"
    )

    if function_description != None:
        cleaned_text = preprocess_text(function_description)
        text = summarize_text(cleaned_text, use_gpt, api_key)
    else:
        text = "Not applicable."
    tokenized_text = text_tokenizer(
        text, max_length=512, truncation=True, return_tensors="pt"
    )

    return tokenized_smiles, tokenized_fasta, tokenized_text


def get_prot_feat(tokenized_fasta, prot_encoder):
    with torch.no_grad():
        encoded_fasta = prot_encoder(**tokenized_fasta)
        encoded_fasta = encoded_fasta.last_hidden_state[:, 0]

    return encoded_fasta


def unsqueeze_everything(
    tokenized_smiles, tokenized_fasta, prot_feat_teacher, tokenized_text, mol_property
):
    tokenized_smiles = tokenized_smiles.unsqueeze(0)
    tokenized_fasta = tokenized_fasta.unsqueeze(0)
    prot_feat_teacher = prot_feat_teacher.unsqueeze(0)
    tokenized_text = tokenized_text.unsqueeze(0)
    mol_property = mol_property.unsqueeze(0)

    return (
        tokenized_smiles,
        tokenized_fasta,
        prot_feat_teacher,
        tokenized_text,
        mol_property,
    )
