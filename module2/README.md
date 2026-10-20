# Parcours NLP et IA — Module 2

Ce dossier rassemble les ressources d'un parcours pratique en traitement automatique du langage naturel (TAL/NLP). Il couvre six modules, des représentations textuelles classiques aux pipelines de données. Chaque sous-dossier contient les supports de cours, les quiz et les travaux pratiques disponibles pour le thème correspondant.

## Parcours

| Étape | Thème | Contenu principal |
|---|---|---|
| 1 | [Fondamentaux du NLP](<./M2_1_Fondamentaux_NLP/>) | Prétraitement, tokenisation, sac de mots, TF-IDF, classification et analyse des erreurs. |
| 2 | [Transformers et embeddings](<./M2_2_Transformers_ Embeddings/>) | Attention et sous-mots, embeddings, recherche sémantique, comparaison avec la recherche par mots, seuil de refus et limites multilingues. |
| 3 | [Fine-tuning, LoRA et QLoRA](<./M2_3_Fine_tuning_LoRA_QLoRA/>) | Adaptation d'un modèle pré-entraîné, entraînement LoRA, évaluation, sauvegarde et bonus QLoRA. |
| 4 | [NER clinique et OCR](<./M2_4_NER_clinique_OCR/>) | Extraction d'entités, négation et contexte, lecture de documents scannés et mesure des erreurs OCR. |
| 5 | [RAG](<./M2_5_RAG/>) | Découpage et recherche documentaire, recherche hybride, réponses avec sources et vérification de leur fidélité. |
| 6 | [Pipelines de données](<./M2_6_Pipelines_de_donnees/>) | Profilage, nettoyage, validation, pseudonymisation, contrôle du k-anonymat et suivi de la qualité. |

L'enchaînement va des bases de représentation et de classification vers des méthodes modernes, puis vers leur intégration dans des systèmes et des flux de données. Les modules peuvent aussi servir de supports indépendants si les prérequis correspondants sont maîtrisés.

## Ressources dans chaque dossier

Les sous-dossiers suivent une organisation commune :

- **Diaporamas** `..._IT_TAL.pptx` et `..._nonIT_TAL.pptx` : supports adaptés aux publics technique et non technique.
- **Atelier non-IT** `Atelier_..._nonIT.docx` : activité ou support d'atelier pour le public non technique.
- **Quiz** `Quiz_..._IT.json` et `Quiz_..._nonIT.json` : questionnaires adaptés aux deux publics.
- **Travaux pratiques** `TP_..._IT_participant.ipynb` et `TP_..._IT_formateur.ipynb` : notebooks de TP pour les participants et corrigés/formateur.

Les quiz JSON contiennent les réponses et leurs explications (`answer` et `expl`) : ne pas les distribuer tels quels aux participants si le corrigé doit rester caché.

## Utiliser les notebooks

Les notebooks sont destinés à être exécutés dans Jupyter ou Google Colab. Ouvrez la version **participant** pour réaliser les exercices, ou la version **formateur** pour disposer des solutions. Exécutez les cellules dans l'ordre et suivez les consignes inscrites dans le notebook : certains exercices demandent de compléter les emplacements `____`.

Les notebooks préparent leurs données et outils dans leurs premières cellules. Consultez ces cellules avant de lancer le TP : certains exercices téléchargent un modèle ou installent des bibliothèques. Le notebook Transformers/embeddings nécessite une connexion Internet lors du premier téléchargement du modèle, mais pas de GPU. Le bonus QLoRA du module 3 requiert un GPU.

## Focus — Transformers et embeddings

Le TP du module 2 est prévu pour environ **1 h 30**. À partir de 16 fiches d'information et de questions d'usagers fictives, il fait construire une recherche qui retrouve les documents pertinents même lorsque la question et le document n'emploient pas les mêmes mots.

Le parcours du notebook couvre :

1. encoder des phrases avec un modèle multilingue et comparer leurs vecteurs par similarité ;
2. encoder un corpus une fois, puis classer les fiches selon leur proximité avec une question ;
3. comparer cette recherche sémantique à une référence TF-IDF par mots ;
4. choisir un seuil pour refuser les demandes auxquelles le corpus ne répond pas ;
5. projeter les embeddings en deux dimensions avec une ACP et regrouper les fiches avec K-means ;
6. observer la tokenisation en sous-mots et tester la robustesse aux fautes, notamment dans un contexte multilingue.

Le TP utilise `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Les exemples sont fictifs ; un GPU n'est pas nécessaire. Les résultats d'un modèle multilingue ne garantissent pas une bonne couverture d'une langue locale : le notebook souligne l'importance de vérifier cette couverture et d'évaluer le modèle sur des données représentatives.

## Quiz en ligne

Le lien vers le quiz interactif et la consultation des résultats est conservé dans [`quiz_link.txt`](./quiz_link.txt). Chaque module fournit également deux fichiers JSON de quiz (IT et non-IT), avec 10 questions et un seuil indiqué de 8.

## Repères sur les thèmes

- **NLP** : construire une première classification de messages et examiner les effets du nettoyage, des représentations et des erreurs.
- **Fine-tuning** : comparer une base TF-IDF à un modèle pré-entraîné adapté avec LoRA ; le notebook présente QLoRA comme bonus nécessitant un GPU.
- **NER clinique et OCR** : extraire des informations de comptes rendus fictifs, tenir compte de la négation, puis mesurer les erreurs introduites par la reconnaissance de documents scannés.
- **RAG** : évaluer la recherche de passages, construire une réponse appuyée sur des sources et en contrôler la fidélité.
- **Pipelines de données** : transformer un registre fictif de vaccination en données mieux contrôlées et documenter les étapes de qualité et de protection.
