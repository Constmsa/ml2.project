import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder

def apriori_rules(customer_basket):

    reader = [eval(item) if isinstance(item, str) else item 
          for item in customer_basket['list_of_goods']]
    
    te = TransactionEncoder()
    te_fit = te.fit(reader).transform(reader)
    transactions_items = pd.DataFrame(te_fit, columns=te.columns_)

    #frequent_itemsets_grocery = apriori(
    #transactions_items, min_support=0.05, use_colnames=True)

    #rules_grocery = association_rules(frequent_itemsets_grocery,
    #                              metric="confidence",
     #                             min_threshold=0.2)
    
    frequent_itemsets_grocery_iter_2 = apriori(
    transactions_items, min_support=0.02, use_colnames=True)

    rules_grocery_iter_2 = association_rules(frequent_itemsets_grocery_iter_2,
                                  metric="confidence",
                                  min_threshold=0.2)

    rules = rules_grocery_iter_2[['antecedents','consequents', 'support','confidence','lift']]

    return rules