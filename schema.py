
from pydantic import BaseModel, Field


class SonarInput(BaseModel):
    # On attend une liste de 60 flottants (C1 à C60)
    features: list[float] = Field(
        ..., 
        example=[0.02, 0.03, 0.04] * 20, # Exemple pour la doc Swagger
        description="Liste des 60 fréquences sonar normalisées entre 0 et 1."
    )