import torch
import torch.nn as nn
import torch.nn.functional as F 

torch.manual_seed(42)

class DTI(nn.Module):
    def __init__(
            self, 
            mol_encoder, 
            prot_encoder,
            text_encoder,
            hidden_dim=1024, 
            mol_dim=768,
            prot_dim=1024,
            text_dim=768
        ):
        
        super().__init__()
        self.mol_encoder = mol_encoder
        self.prot_encoder = prot_encoder
        self.text_encoder = text_encoder
        self.lambda_ = torch.ones(1)
        
        self.molecule_align = nn.Sequential(
            nn.LayerNorm(mol_dim),
            nn.Linear(mol_dim, hidden_dim, bias=False)
        )
        
        self.protein_align_teacher = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Linear(1024, hidden_dim, bias=False)
        )
        
        self.protein_align_student = nn.Sequential(
            nn.LayerNorm(prot_dim),
            nn.Linear(prot_dim, hidden_dim, bias=False)
        )
        
        self.text_align = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim, bias=False)
        )
        
        self.property_align = nn.Sequential(
            nn.LayerNorm(209),
            nn.Linear(209, hidden_dim, bias=False)
        )
        
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim * 2)
        self.fc3 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.cls_out = nn.Linear(hidden_dim, 1)
        
    def forward(self, SMILES, FASTA, prot_feat_teacher, text, property):
        mol_feat = self.mol_encoder(**SMILES).last_hidden_state[:, 0]
        prot_feat = self.prot_encoder(**FASTA).last_hidden_state[:, 0]
        
        mol_feat = self.molecule_align(mol_feat).squeeze(1)
        prot_feat = self.protein_align_student(prot_feat)
        prot_feat_teacher = self.protein_align_teacher(prot_feat_teacher).squeeze(1)
        
        text_feat = self.text_encoder(**text).last_hidden_state[:, 0]
        text_feat = self.text_align(text_feat)
        
        property_feat = self.property_align(property).unsqueeze(0)
        
        lambda_ = torch.sigmoid(self.lambda_)           
        merged_prot_feat = lambda_ * prot_feat + (1 - lambda_) * prot_feat_teacher
        
        x_drug = torch.cat([mol_feat, property_feat], dim=1)
        x_target = torch.cat([merged_prot_feat, text_feat], dim=1)
            
        x = torch.cat([x_drug, x_target], dim=1)
        x = F.dropout(F.gelu(self.fc1(x)), 0.1)
        x = F.dropout(F.gelu(self.fc2(x)), 0.1)
        x = F.dropout(F.gelu(self.fc3(x)), 0.1)
        
        cls_out = self.cls_out(x).squeeze(-1)
        
        return cls_out, self.lambda_.mean()
