**Rakuten Project KPIs**

Finalized model will be be deployed into production environment and exposed to users on the live website. Model will 
make predictions based on user input data when creating a new posting, and pre-populate the category of the 
posting. Users will be allowed to modify the category if needed.

Key performance metrics: 
- Model latency for making predictions
- Number of requests per minute that can be handled by the server
- Number of requests per second allowed to individual users (security)
- Recall performance to monitor the share of actual categories that were correctly predicted (globally and per group)
    - Measured by user changes to predicted categories that were proposed
- Request and monitor feedback from users on usefulness of predicted categories
- A/B Test time improvement in posting creation for users with and without the feature


Key assumptions monitoring:
- Data drift across groups including shifts in total share between groups 
    - e.g. over time one group becomes larger relative to others compared to its share in the training data




