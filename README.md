Who might care about this problem and why?

Ability to predict and forecast insurance claims is an important business requirement for insurance firms. Insurance claim prediction models can help actuaries better evaluate risks associated with insurance products. 
Even though this problem is related to the vehicle insurance segment, it has multiple stakeholders involved directly or indirectly – 
1.	Department of transportation – Vehicle insurance claims might be directly related to traffic incidents. Understanding risk factors for vehicle insurance claims can also help mitigate traffic accidents 
2.	Healthcare – Vehicle insurance claims can be conflated with health insurance claims to forecast requirement of healthcare resources
3.	Automobile makers – Helps them improve safety in cars and forecast vehicle repair numbers

Why might this problem be challenging?

One of the biggest challenges is posed by the large number of variables, so we have to avoid succumbing to the curse of dimensionality. For this particular problem, the predictor variable names were anonymized to maintain the privacy of customers. This further increased the complexity as ignorance of data definitions prevented us from engineering the features. The skewness of the response variable is another challenge, which is overcome by down-sampling the training dataset. This skew has to be maintained in the predictions other-wise the customers of the insurance firms might face the ramifications. If the model predicts a higher that actual rate of insurance claims, the additional burden would be borne by the customers in the form of higher premiums. We also have to be careful while modelling to make sure that the false positive rate is not exceedingly high

What other problems resemble this problem?

1.	For Software-as-a-Service (SaaS) companies - Account churn is a major business problem for SaaS companies. The problem is similar as there are a large number of predictor variables across multiple categories such as – 
a.	Account related predictors - $ value of account, # of subscriptions purchased
b.	Company information – Number of employees, geography, revenue
c.	Usage metrics – Number of visitors, active users
Additionally, the response variable could be equally skewed if the account churn rate is small (ie. the business isn’t losing a lot of customers)

2.	For fault/defect detection – Faults or defects within manufactured products cause wastage, increase maintenance costs and can potentially have legal implications. The related problem is to predict defects based on a set of predictor variables. Usually the rate of defects is quite low, which leads to a skewed response, just like the insurance problem.
