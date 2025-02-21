#decision tree
- predicting US voting preference with knn, k = 1, test error obtained = 0.065
- ![Alt text](https://github.com/user-attachments/files/18913456/Q1.3.pdf)
- Finding best k for knn, on data/ccdata.pkl, contains a subset of Statistics Canada’s 2019 Survey of Financial Security; we’re
predicting whether a family regularly carries credit card debt, based on a bunch of demographic and financial
information about them. Using 10-fold cross-validation we found:
-![Alt text](https://github.com/user-attachments/files/18913462/q2.3.pdf)
- Explored role of lapace smoothing in Naive Bayes, after implementing naive bayes the testing error went from 0.188 -> 0.187
- Exploring the power of Random forest (50 trees, and infinite depth)
     
     n = 264, d = 10
     
  Decision tree info gain
     
       Training error: 0.000
       Testing error: 0.367
     
  Random tree
  
      Training error: 0.148
      Testing error: 0.470
    Random Forest
  
      Training error: 0.000
      Testing error: 0.186
- kmeans with (k = 4)
- before:![Alt text](https://github.com/user-attachments/assets/fb4b90a3-aa42-497b-abfc-ab1b9ea55d52)
- after:![Alt text](https://github.com/user-attachments/assets/8a954c78-31c3-4001-915c-020519824189)
