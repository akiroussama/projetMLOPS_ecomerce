\# Data Contract \& Dictionnaire des Features



Ce document formalise le schéma de données attendu et le dictionnaire des variables pour le projet de classification E-commerce, en réponse au ticket #9.



\## 1. Dictionnaire des Features (Variables)



| Variable | Description métier |

| :--- | :--- |

| \*\*Unnamed: 0\*\* | Index de la ligne, sert d'identifiant technique pour lier les features (X) à la variable cible (Y). |

| \*\*designation\*\* | Titre commercial ou nom court du produit. |

| \*\*description\*\* | Texte descriptif détaillé du produit. |

| \*\*productid\*\* | Identifiant unique du produit. |

| \*\*imageid\*\* | Identifiant de l'image associée au produit (permet de retrouver le fichier image correspondant). |

| \*\*prdtypecode\*\* | \*Target\* (Cible) - Code entier représentant la catégorie du produit que le modèle devra prédire. |



\## 2. Data Contract (Contraintes du Schéma)



Pour que les pipelines de traitement et d'entraînement s'exécutent sans erreur, les fichiers d'entrée (`X\_train`, `X\_test`, `Y\_train`) doivent impérativement respecter les règles suivantes :



| Colonne | Type attendu | Nullable (Peut être vide ?) | Contraintes supplémentaires |

| :--- | :--- | :--- | :--- |

| \*\*Unnamed: 0\*\* | `Integer` (int64) | ❌ Non | Valeur unique pour chaque ligne. |

| \*\*designation\*\* | `String` (Texte) | ❌ Non | Chaîne de caractères obligatoire. |

| \*\*description\*\* | `String` (Texte) | ✅ \*\*Oui\*\* | La colonne peut contenir du texte très long ou des valeurs `NaN` (vides). Le code devra gérer ces valeurs manquantes. |

| \*\*productid\*\* | `Integer` (int64) | ❌ Non | Identifiant numérique. |

| \*\*imageid\*\* | `Integer` (int64) | ❌ Non | Doit pointer vers une image valide téléchargée dans le dossier brut. |

| \*\*prdtypecode\*\* | `Integer` (int64) | ❌ Non | Présent uniquement dans le fichier cible (`Y\_train`). |



\*Note de formatage : Les fichiers CSV doivent utiliser la virgule `,` comme séparateur de colonnes.\*

