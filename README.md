# Credit Card Fraud Detection

<ol>

<li>Came across this mocked-up dataset of customer transactions at [Capital One Recruitment Challenge](https://github.com/CapitalOneRecruiting/DS).

<li>The unbalanced dataset is comprised of artificial customer transactions with a few outlier cases where fraud was detected. There's only ~1.6% fraudulent cases.</li>

<li>Our primary goal is to successfully predict whether a transaction is Fraudulent or not, and avoid Type-II errors as much as possible as in most sensitive classification problems: we'll try not to point accusatory-fingers at genuine-transactions :joy:.</li>

<li>The secondary goal is to identify interesting anomalies in the transactions like multi-swipes, reversal of suspicious transactions, etc. by performing exploratory-data-analysis.</li>

<li>Most numerical-fields seem to follow Power-law distributions rather than Gaussian distributions.</li>

<li>We'll engineer some time-dependent categorical features by parsing the datetime fields, exclude the fields which have just one categorical value (makes no sense keeping these around :unamused:), and also create a new feature to indicate if credit-card-CVV is wrongly entered.</li>

<li>Baseline classifiers chosen are Logistic Regression, SVM, Random Forest, Isolated Forest.</li>

<li>Performance is kinda poor on these Baseline models: Accuracy, precision, and recall vary greatly across the models.</li>

<li>Moving on Gradient-Boosting models, Light Gradient Boosting is known to perform well on sparse datasets.</li>

<li>Final accuracy achieved hovers around 98%, and recall is approximately 99.99% indicating that False-Negatives are absolutely minimal.</li>

</ol>