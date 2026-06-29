from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyStep:
    log_prob: torch.Tensor
    value: torch.Tensor
    reward: float
    entropy: torch.Tensor


class MaskedPolicyNetwork(nn.Module):
    """
    Réseau acteur-critique léger pour le CVRP.

    Il produit :
    - un score pour chaque nœud ;
    - une valeur estimée de l'état courant.

    Les actions invalides sont ensuite masquées avant l'échantillonnage.
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 128,
    ):
        super().__init__()

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_head = nn.Linear(hidden_dim, 1)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        node_features : [n_nodes, feature_dim]
        action_mask   : [n_nodes], True pour action valide, False sinon
        """
        embeddings = self.node_encoder(node_features)

        logits = self.actor_head(embeddings).squeeze(-1)

        # Masquer les actions invalides.
        masked_logits = logits.masked_fill(~action_mask, -1e9)

        # Valeur globale de l'état : moyenne des embeddings.
        global_embedding = embeddings.mean(dim=0)
        value = self.value_head(global_embedding).squeeze(-1)

        return masked_logits, value


def masked_categorical_sample(
    logits: torch.Tensor,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """
    Échantillonne une action selon une distribution catégorielle masquée.
    """
    probs = F.softmax(logits, dim=-1)
    distribution = torch.distributions.Categorical(probs=probs)

    action = distribution.sample()
    log_prob = distribution.log_prob(action)
    entropy = distribution.entropy()

    return int(action.item()), log_prob, entropy


def masked_categorical_greedy(
    logits: torch.Tensor,
) -> int:
    """
    Choisit l'action la plus probable.
    """
    return int(torch.argmax(logits).item())