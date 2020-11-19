import os

import pandas as pd

class tagged_data_reader():
    def __init__(self,read_details = False):
        self.read_details = read_details
    disputed = ['DB00603', 'DB00641', 'DB00798', 'DB00968', 'DB01033', 'DB01222', 'DB14509']

    def read_all(self,remove_disputed=True,read_who=True,read_smc=True,read_safeFetus=True,read_Eltonsy_et_al=True):
        l = []
        if read_who:
            l.append(self.read_who(remove_disputed=False))
        if read_smc:
            l.append(self.read_smc(remove_disputed=False))
        if read_safeFetus:
            l.append(self.read_safeFetus(remove_disputed=False))
        if read_Eltonsy_et_al:
            l.append(self.read_Eltonsy_et_al(remove_disputed=False))


        ans =pd.concat(l) #, self.read_Eltonsy_et_al(remove_disputed=False)
        print('num drugs before removing duplicates:',len(ans))

        #ans = ans.drop_duplicates(keep='first',subset='index')
        ans = ans.loc[~ans.index.duplicated(keep='first')]

        print('num drugs after removing duplicates:',len(ans))
        if remove_disputed:
            ans = ans.loc[~ans.index.isin(self.getDisputed())]
        # else:
        #     ans.loc[ans.drugBank_id.isin(self.disputed),'preg_class'] = 'Limited'

        #ans.loc[ans.drugBank_id.isin(disputed),'preg_class'] = 'Safe'
        print('Total:', len(ans))
        return ans

    def read_who(self,remove_disputed=True):
        tagged_drugs_path=os.path.join('data', 'Polifka et al. classification.xlsx')
        ans = pd.read_excel(tagged_drugs_path)
        ans = ans.dropna(subset=['drugBank_id'])
        ans = ans[ans['comment'].isna()]
        ans = ans[ans['ophth']!='Yes']
        ans=ans[['drugBank_id','Polifka et al. classification']]
        ans.columns=['drugBank_id', 'preg_class']
        if not self.read_details:
            ans.preg_class= ans.preg_class.replace('Suggestive Risk', 'Known Risk')
            ans.preg_class = ans.preg_class.replace('Known Risk', 'Limited')
            ans.preg_class = ans.preg_class.replace('Little to No Risk', 'Safe')
            ans=ans[ans.preg_class!='Insufficient Information']
            ans=ans[ans.preg_class!='Insufficient Information due to decompose']
        ans = ans.drop_duplicates(subset=['drugBank_id'])
        ans = ans.set_index('drugBank_id')
        if remove_disputed:
            ans = ans.loc[~ans.index.isin(self.getDisputed())]
        print('WHO:',len(ans))
        return ans


    def read_smc(self,filter_sys_only=False,remove_disputed=True):
        ans = pd.read_excel(
            os.path.join('data', 'Zerifin classification.xlsx'))
        ans = ans.dropna(subset=['drugBank_id'])
        print('exact match and connected to drugbank', len(ans))
        if filter_sys_only:
            ans = ans[ans[r'Systemic\Not systemic'] == 'sys']

        #classification_field = 'My Classification'
        classification_field = 'Final_clas'

        ans = ans[['drugBank_id', classification_field]]
        # ans[classification_field] = ans[classification_field].str.replace('Limited-Manual',
        #                                                                                         'Limited')
        # ans[classification_field] = ans[classification_field].str.replace('Limited Week-Manual',
        #                                                                                         'Limited')
        # ans[classification_field] = ans[classification_field].str.replace('Safe-Manual', 'Safe')
        # ans[classification_field] = ans[classification_field].str.replace('Limited Week',
        #                                                                                         'Limited')
        if not self.read_details:
            ans.loc[ans[classification_field] == 'Week dependent + Dosage dependent',classification_field]='Limited'
            ans[classification_field] = ans[classification_field].str.replace('Dosage dependent', 'Limited')
            ans[classification_field] = ans[classification_field].str.replace('High Risk', 'Limited')
            ans[classification_field] = ans[classification_field].str.replace('Low Risk', 'Safe')
            ans[classification_field] = ans[classification_field].str.replace('Week dependent', 'Limited')
            ans = ans[ans[classification_field] != 'Not enough information']
            ans = ans[ans[classification_field] != 'Not intended for pregnancy use']
        ans.columns = ['drugBank_id', 'preg_class']
        ans = ans.set_index('drugBank_id')
        if remove_disputed:
            ans = ans.loc[~ans.index.isin(self.disputed)]

        print('SMC:',len(ans))
        return ans

    def read_Eltonsy_et_al(self,remove_disputed=False):
        ans = pd.read_excel(
            os.path.join('pickles', 'data', 'preg', 'Eltonsy_et_al_table1.xlsx'))
        ans=ans[ans['Comment'].isna()]
        ans=ans[~ans.drugBank_id.isna()]
        ans=ans[['drugBank_id']]
        ans['preg_class']="Limited"
        assert ans.drugBank_id.nunique()==ans.drugBank_id.count(),'duplicated found'
        ans=ans.set_index('drugBank_id')
        if remove_disputed:
            ans = ans.loc[~ans.index.isin(self.disputed)]
        print('Eltonsy:',len(ans))
        return ans

    def read_safeFetus(self,remove_disputed=False,convert_binary=True):
        ans = pd.read_excel(
            os.path.join('pickles', 'data', 'preg', 'Teratogenicity scores extracted from SafeFetus (AS).xlsx'))
        if convert_binary:
            ans = ans[ans['preg_risk_cat']!='C']
        ans=ans[['drugBank_id','preg_risk_cat']]
        ans.columns = ['drugBank_id','preg_class']
        ans = ans.drop_duplicates(subset=['drugBank_id'])
        if convert_binary:
            ans.preg_class= ans.preg_class.replace('A', 'Safe')
            ans.preg_class = ans.preg_class.replace('B', 'Safe')
            ans.preg_class = ans.preg_class.replace('D', 'Limited')
            ans.preg_class = ans.preg_class.replace('X', 'Limited')
        ans=ans.set_index('drugBank_id')
        if remove_disputed:
            ans = ans.loc[~ans.index.isin(self.disputed)]
        print('SafeFetus:',len(ans))

        return ans

    def read_SMC_corona(self):
        ans = pd.read_excel(
            os.path.join('pickles', 'data', 'preg', 'SMC_corona_drugs.xlsx'))
        ans=ans[['drugBank_id','preg_class']]
        assert len(ans.drugBank_id) == ans.drugBank_id.nunique()
        ans = ans[~ans.preg_class.str.contains('No Data')]
        ans['preg_class']=ans['preg_class'].str.replace('*','')
        ans=ans.set_index('drugBank_id')

        print('SMC corona:',len(ans))

        return ans


    def getDisputed(self):
        return self.disputed
