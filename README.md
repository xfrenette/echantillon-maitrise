Échantillons de code
===

Je présente dans ce dépôt une sélection de scripts développés durant mon 
mémoire de maitrise. Ils ont été sélectionnés pour démontrer la variété de 
mes compétences en programmation.

Il ne s'agit que d'une *infime* partie de ce qui a été développé, mais ces
extraits sont représentatifs du reste. Les scripts présentés ici ont été
sortis de leur contexte, ils ne peuvent donc pas être exécutés directement
puisqu'ils nécessitent l'accès à d'autres modules qui ne se trouvent pas
dans ce dépôt.

Sont également inclus des tests unitaires développés pour ces scripts. Je 
les ai mis dans le même dossier, mais, dans la structure originale, ils se 
trouvaient dans un dossier séparé. La structure de ce dépôt ne correspond
d'ailleurs pas à la structure originale.

## Dossier `operations-numpy`

Ce dossier contient un script que j'ai développé et qui démontre une 
utilisation avancée des opérations sur les matrices 
avec Numpy. La fonction `normals_kl_div` permet de calculer, à partir d'une 
liste de normales multivariées (chacune définie par son espérance et 
sa matrice de covariance)
[la divergence de Kullback-Leibler](https://fr.wikipedia.org/wiki/Divergence_de_Kullback-Leibler)
entre chaque paire. La fonction n'a besoin d'aucune boucle, 
utilisant seulement les opérations matricielles et la technique de 
"broadcasting" de Numpy. J'ai développé cette fonction moi-même, utilisant 
seulement quelques sources citées dans les commentaires.

Le script contient également une autre fonction pour calculer la
[divergence de Jensen-Shannon](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence)
entre des normales multivariées. Il n'a pas été possible d'éviter 
l'utilisation d'une boucle pour cette fonction.

## Dossier `multiprocessus`

Ce dossier contient une application qui a été développée pour calculer 
différents scores de performance sur un modèle de traduction.

Dans mon mémoire, j'ai eu à comparer plusieurs modèles. Différentes 
mesures de performances (BLEU, chrF++ et BERTScore) devaient être calculées 
pour chacun. De plus, chaque modèle devait être testé sur plusieurs 
documents différents.

Vu les limites technologiques de nos serveurs, exécuter de façon 
séquentielle les différentes tâches (traduction, évaluation BLEU, chrF++, 
BERTScore) pour chaque modèle et sur chaque document aurait pris un temps 
déraisonnable.

L'application présentée ici utilise des techniques de parallélisme par 
processus (multiprocessing) pour exécuter les différentes tâches en 
parallèle sur plusieurs processeurs (CPU). Un premier processus traduit un 
ensemble de phrases. Lorsque terminé, il les dépose dans une file d'attente 
et démarre immédiatement une autre tâche de traduction. Parallèlement, cinq 
processus évaluent le score BLEU sur les traductions produites, et cinq autres 
évaluent le score chrF++.

Cette distribution parallèle des tâches a permis de réduire par 10 le temps 
d'évaluation.

## Dossier `math-en-python`

Dans mon mémoire, j'ai eu, à plusieurs reprises, à implémenter en code des 
méthodes uniquement décrites par des équations mathématiques.

Par exemple, j'ai eu à déterminer la similarité entre deux documents. La 
littérature scientifique propose différentes méthodes mathématiques pour 
calculer ce type de similarité. Le script de ce dossier présente des 
implémentations en Python de quelques-unes de ces techniques que j'ai 
développées à partir de leur définition mathématique.

## Dossier `bash`

Automatiser les processus sur un serveur Linux demande parfois de créer des 
scripts Shell. Je présente dans ce dossier des scripts Bash qui permettent 
de télécharger des données, de les prétraiter et de démarrer l'application 
d'évaluation d'un modèle préentrainé.
