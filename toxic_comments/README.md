=======identity_hate
              precision    recall  f1-score   support

         0.0       0.99      1.00      1.00     15811
         1.0       0.62      0.18      0.28       147

    accuracy                           0.99     15958
   macro avg       0.81      0.59      0.64     15958
weighted avg       0.99      0.99      0.99     15958

=======insult
              precision    recall  f1-score   support

         0.0       0.98      0.99      0.98     15167
         1.0       0.73      0.56      0.64       791

    accuracy                           0.97     15958
   macro avg       0.86      0.78      0.81     15958
weighted avg       0.97      0.97      0.97     15958

=======obscene
              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99     15122
         1.0       0.85      0.76      0.80       836

    accuracy                           0.98     15958
   macro avg       0.92      0.88      0.90     15958
weighted avg       0.98      0.98      0.98     15958

=======severe_toxic
              precision    recall  f1-score   support

         0.0       0.99      1.00      1.00     15810
         1.0       0.57      0.26      0.36       148

    accuracy                           0.99     15958
   macro avg       0.78      0.63      0.68     15958
weighted avg       0.99      0.99      0.99     15958

=======threat
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     15921
         1.0       0.00      0.00      0.00        37

    accuracy                           1.00     15958
   macro avg       0.50      0.50      0.50     15958
weighted avg       1.00      1.00      1.00     15958

=======toxic
              precision    recall  f1-score   support

         0.0       0.96      0.99      0.97     14478
         1.0       0.84      0.60      0.70      1480

    accuracy                           0.95     15958
   macro avg       0.90      0.79      0.84     15958
weighted avg       0.95      0.95      0.95     15958

with 80/20 split:
=======identity_hate
              precision    recall  f1-score   support

         0.0       0.99      1.00      1.00     31621
         1.0       1.00      0.01      0.01       294

    accuracy                           0.99     31915
   macro avg       1.00      0.50      0.50     31915
weighted avg       0.99      0.99      0.99     31915

=======insult
              precision    recall  f1-score   support

         0.0       0.98      0.99      0.98     30301
         1.0       0.75      0.54      0.63      1614

    accuracy                           0.97     31915
   macro avg       0.86      0.77      0.80     31915
weighted avg       0.96      0.97      0.96     31915

=======obscene
              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99     30200
         1.0       0.85      0.76      0.80      1715

    accuracy                           0.98     31915
   macro avg       0.92      0.88      0.90     31915
weighted avg       0.98      0.98      0.98     31915

=======severe_toxic
              precision    recall  f1-score   support

         0.0       0.99      1.00      1.00     31594
         1.0       0.59      0.32      0.41       321

    accuracy                           0.99     31915
   macro avg       0.79      0.66      0.70     31915
weighted avg       0.99      0.99      0.99     31915

=======threat
              precision    recall  f1-score   support

         0.0       1.00      1.00      1.00     31841
         1.0       0.00      0.00      0.00        74

    accuracy                           1.00     31915
   macro avg       0.50      0.50      0.50     31915
weighted avg       1.00      1.00      1.00     31915

=======toxic
              precision    recall  f1-score   support

         0.0       0.96      0.99      0.97     28859
         1.0       0.88      0.58      0.70      3056

    accuracy                           0.95     31915
   macro avg       0.92      0.78      0.84     31915
weighted avg       0.95      0.95      0.95     31915

with 80/20 split with class weighting:
=======identity_hate
              precision    recall  f1-score   support

         0.0       1.00      0.90      0.95     31621
         1.0       0.08      0.93      0.14       294

    accuracy                           0.90     31915
   macro avg       0.54      0.91      0.54     31915
weighted avg       0.99      0.90      0.94     31915

=======insult
              precision    recall  f1-score   support

         0.0       0.99      0.90      0.94     30301
         1.0       0.30      0.85      0.45      1614

    accuracy                           0.89     31915
   macro avg       0.65      0.87      0.69     31915
weighted avg       0.96      0.89      0.92     31915

=======obscene
              precision    recall  f1-score   support

         0.0       0.99      0.95      0.97     30200
         1.0       0.52      0.89      0.65      1715

    accuracy                           0.95     31915
   macro avg       0.76      0.92      0.81     31915
weighted avg       0.97      0.95      0.96     31915

=======severe_toxic
              precision    recall  f1-score   support

         0.0       1.00      0.95      0.98     31594
         1.0       0.17      0.98      0.29       321

    accuracy                           0.95     31915
   macro avg       0.59      0.97      0.63     31915
weighted avg       0.99      0.95      0.97     31915

=======threat
              precision    recall  f1-score   support

         0.0       1.00      0.89      0.94     31841
         1.0       0.02      0.89      0.04        74

    accuracy                           0.89     31915
   macro avg       0.51      0.89      0.49     31915
weighted avg       1.00      0.89      0.94     31915

=======toxic
              precision    recall  f1-score   support

         0.0       0.98      0.88      0.92     28859
         1.0       0.41      0.82      0.55      3056

    accuracy                           0.87     31915
   macro avg       0.70      0.85      0.74     31915
weighted avg       0.92      0.87      0.89     31915

with 80/20 split with smote:
=======identity_hate
              precision    recall  f1-score   support

         0.0       1.00      0.96      0.98     31621
         1.0       0.13      0.59      0.21       294

    accuracy                           0.96     31915
   macro avg       0.56      0.78      0.59     31915
weighted avg       0.99      0.96      0.97     31915

with 80/20 split with borderline smote:
=======identity_hate
              precision    recall  f1-score   support

         0.0       1.00      0.95      0.97     31621
         1.0       0.11      0.69      0.19       294

    accuracy                           0.95     31915
   macro avg       0.55      0.82      0.58     31915
weighted avg       0.99      0.95      0.96     31915

with 80/20 split with borderline smote, undersampling, and XGBoost
=======identity_hate
              precision    recall  f1-score   support

         0.0       1.00      0.99      0.99     31621
         1.0       0.43      0.55      0.48       294

    accuracy                           0.99     31915
   macro avg       0.71      0.77      0.74     31915
weighted avg       0.99      0.99      0.99     31915

 true     0 [31406   215]
          1 [  133   161]
               0     1   (pred)

with 80/20 split with borderline smote, undersampling (TomekLinks), and XGBoost
=======identity_hate
               precision    recall  f1-score   support

         0.0       0.99      1.00      1.00     31621
         1.0       0.52      0.46      0.49       294

    accuracy                           0.99     31915
   macro avg       0.76      0.73      0.74     31915
weighted avg       0.99      0.99      0.99     31915

[[31497   124]
 [  159   135]]

with 80/20 split with borderline smote, class weights, and XGBoost
=======identity_hate
              precision    recall  f1-score   support

         0.0       0.99      1.00      1.00     31621
         1.0       0.52      0.45      0.48       294

    accuracy                           0.99     31915
   macro avg       0.76      0.72      0.74     31915
weighted avg       0.99      0.99      0.99     31915

[[31497   124]
 [  162   132]]

with 80/20 split with ensemble (class weights and unweighted) and simple averaging ****
identity_hate
              precision    recall  f1-score   support

         0.0       1.00      0.98      0.99     31621
         1.0       0.25      0.64      0.36       294

    accuracy                           0.98     31915
   macro avg       0.62      0.81      0.67     31915
weighted avg       0.99      0.98      0.98     31915

[[31056   565]
 [  107   187]]
insult
              precision    recall  f1-score   support

         0.0       0.98      0.98      0.98     30301
         1.0       0.60      0.70      0.64      1614

    accuracy                           0.96     31915
   macro avg       0.79      0.84      0.81     31915
weighted avg       0.96      0.96      0.96     31915

[[29545   756]
 [  487  1127]]
obscene
              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99     30200
         1.0       0.77      0.82      0.79      1715

    accuracy                           0.98     31915
   macro avg       0.88      0.90      0.89     31915
weighted avg       0.98      0.98      0.98     31915

[[29771   429]
 [  302  1413]]
severe_toxic
              precision    recall  f1-score   support

         0.0       1.00      0.98      0.99     31594
         1.0       0.30      0.87      0.45       321

    accuracy                           0.98     31915
   macro avg       0.65      0.93      0.72     31915
weighted avg       0.99      0.98      0.98     31915

[[30951   643]
 [   41   280]]
threat
              precision    recall  f1-score   support

         0.0       1.00      0.98      0.99     31841
         1.0       0.07      0.51      0.12        74

    accuracy                           0.98     31915
   macro avg       0.53      0.75      0.56     31915
weighted avg       1.00      0.98      0.99     31915

[[31340   501]
 [   36    38]]
toxic
              precision    recall  f1-score   support

         0.0       0.96      0.98      0.97     28859
         1.0       0.75      0.66      0.70      3056

    accuracy                           0.95     31915
   macro avg       0.86      0.82      0.84     31915
weighted avg       0.94      0.95      0.95     31915

[[28202   657]
 [ 1046  2010]]

with 80/20 split with ensemble (class weights and unweighted) and weighted voting (class weight model has 2 votes)
 identity_hate
              precision    recall  f1-score   support

         0.0       1.00      0.89      0.94     31621
         1.0       0.07      0.91      0.14       294

    accuracy                           0.89     31915
   macro avg       0.54      0.90      0.54     31915
weighted avg       0.99      0.89      0.94     31915

[[28231  3390]
 [   26   268]]
insult
              precision    recall  f1-score   support

         0.0       0.99      0.91      0.95     30301
         1.0       0.33      0.83      0.47      1614

    accuracy                           0.91     31915
   macro avg       0.66      0.87      0.71     31915
weighted avg       0.96      0.91      0.92     31915

[[27568  2733]
 [  270  1344]]
obscene
              precision    recall  f1-score   support

         0.0       0.99      0.96      0.98     30200
         1.0       0.56      0.88      0.68      1715

    accuracy                           0.96     31915
   macro avg       0.78      0.92      0.83     31915
weighted avg       0.97      0.96      0.96     31915

[[29022  1178]
 [  214  1501]]
severe_toxic
              precision    recall  f1-score   support

         0.0       1.00      0.96      0.98     31594
         1.0       0.19      0.96      0.32       321

    accuracy                           0.96     31915
   macro avg       0.60      0.96      0.65     31915
weighted avg       0.99      0.96      0.97     31915

[[30297  1297]
 [   13   308]]
threat
              precision    recall  f1-score   support

         0.0       1.00      0.90      0.95     31841
         1.0       0.02      0.91      0.04        74

    accuracy                           0.90     31915
   macro avg       0.51      0.90      0.49     31915
weighted avg       1.00      0.90      0.94     31915

[[28612  3229]
 [    7    67]]
toxic
              precision    recall  f1-score   support

         0.0       0.98      0.86      0.92     28859
         1.0       0.39      0.84      0.53      3056

    accuracy                           0.86     31915
   macro avg       0.68      0.85      0.72     31915
weighted avg       0.92      0.86      0.88     31915

[[24801  4058]
 [  480  2576]]