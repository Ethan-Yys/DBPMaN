import pandas as pd
data = pd.read_csv('./Taobao_data.csv')
reviews_info = data[['User_Id','Item_Id','Behavior_type','Timestamp']]
item_info = data[['Item_Id','Category_Id']].drop_duplicates()
reviews_info.to_csv('./reviews-info',header=None,index=False,sep='\t')
item_info.to_csv('./item-info',header=None, index=False,sep='\t')
print('gen_finished')
