import os, datetime
import pandas as pd, numpy as np

class tab():
    def __init__(
        self,
        df,
        catvars,
        varlab=None,
        vallab=None,
        col=None):
        self.df=df
        self.catvars=catvars
        self.varlab=varlab
        self.vallab=vallab
        self.col=col
        
        self.cat_table=self.tabcat()
        self._cat_table=self.tabcat()
        self._overall=self._overall()
    def missing_count(self):
        missing_table=pd.concat([
            self.df[self.catvars].isna().sum(),
            (self.df[self.catvars].isna().sum()*100/self.df.shape[0]).round(2)
        ], axis=1, keys=['#missing', '%missing'])
        return missing_table
    
    def tabcat(self, 
               n_decimals=1):
        if self.col==None:
            self.col='_dummy'
            self.df[self.col]=1
        
        # cal freq
        count=pd.concat([
                pd.crosstab(self.df[r], self.df[self.col], dropna=False) 
                for r in self.catvars], 
            keys=self.catvars)
        # cal percentages
        perct=pd.concat([
                pd.crosstab(self.df[r], self.df[self.col], dropna=False, normalize='columns') 
                for r in self.catvars], 
            keys=self.catvars)        
        perct=round(perct*100, n_decimals)

        # make table (consider sort columns)
        # !consideration of ordering of object (string) col
        table=pd.concat([count, perct], axis=1).sort_index(axis=1)
        col_index=[]
        for col_l2 in table.columns.get_level_values(0).unique():
            col_index.extend([
                (self.col, col_l2, 'n'), (self.col, col_l2, '%')
            ])
        if self.col=='_dummy':
            table.columns=['n','%']
        else:
            table.columns=pd.MultiIndex.from_tuples(col_index)
        table.index.names=[None, None]
        
        MISS=self.missing_count()
        MISS=MISS.loc[MISS['#missing']>0]
        # print missing for caution
        if len(MISS)==0:
            MISS_print=f'There is NO missing in {self.catvars}!'
        else:
            MISS_print=f'Has missing in {self.catvars}, vars miss:\n{MISS.to_string()}'
        print(MISS_print)
        
        return table

    def add_label(self):
        if self.varlab is None or self.vallab is None:
            raise Exception("Must have label")
        
        # label values
        if self.vallab is not None:
            # label columns only apply when col!=_dummy
            if self.col!='_dummy':
                cl1, cl2, cl3=zip(*self.cat_table.columns) # cl stand for: column level
                col_index=[
                    (l1, self.vallab[l1][l2], l3) if l1 in self.vallab else (l1,l2,l3)
                    for l1,l2,l3 in zip(cl1, cl2, cl3)
                ]
                self.cat_table.columns=pd.MultiIndex.from_tuples(col_index)
                
            # label rows level
            rl1, rl2=zip(*self.cat_table.index)
            row_index=[
                (l1, self.vallab[l1][l2]) if l1 in self.vallab else (l1, l2) 
                for l1,l2 in zip(rl1, rl2)
            ]
            self.cat_table.index=pd.MultiIndex.from_tuples(row_index)
        
        # label variable
        if self.varlab is not None: 
            to_rename_row={r: self.varlab[r] for r in self.catvars if r in self.varlab}
            to_rename_col={self.col: self.varlab[self.col]} if self.col in self.varlab else {}
            self.cat_table.rename(columns=to_rename_col, inplace=True)
            self.cat_table.rename(index=to_rename_row, inplace=True)
        
        return self.cat_table
    
    def drop_bin(self, 
               binvars, 
               no_value):
        '''
        dropbin must be applied in the last step
        '''
        assert all([x in self.catvars for x in binvars])
        indexl1=self.cat_table.index.get_level_values(0)
        if all([x not in indexl1 for x in binvars]):
            # update binvars
            binvars=[self.varlab[x] for x in binvars]
        
        row_removes=[
            (l1, no_value) for l1 in binvars
        ]
        
        self.cat_table=self.cat_table.drop(row_removes, axis=0)

        return self.cat_table
    
    def _overall(self):
        # using the same approach with above function
        self.df['_i']=1 # for counting
        count=pd.concat([
                pd.crosstab(self.df[r], self.df['_i'], dropna=False) 
                for r in self.catvars], 
            keys=self.catvars)
        # cal percentages
        perct=pd.concat([
                pd.crosstab(self.df[r], self.df['_i'], dropna=False, normalize='columns') 
                for r in self.catvars], 
            keys=self.catvars)
        perct=round(perct*100, 1)

        _overall=pd.concat([count, perct], axis=1)
        col_index=pd.MultiIndex.from_tuples([('', 'All', 'n'), ('', 'All', '%')])
        _overall.columns=col_index

        return _overall

    def add_overall(self,
                     position):
        '''
        !updated on: 20240505
        note: must used before drop_bin method
        drop_bin should be used at last because it change whole index of the table
        '''
        assert position in ['head', 'tail']
        if self.col=='_dummy': raise Exception('this only apply with group column')
        
        # merging using index
        if self._overall.index.size==self.cat_table.index.size:
            # this case without using drop_bin
            self._overall.index=self.cat_table.index
            if position=='tail':
                self.cat_table=pd.concat(
                    [self.cat_table, self._overall], axis=1
                )
            else:
                self.cat_table=pd.concat(
                    [self._overall, self.cat_table], axis=1
                )

        return self.cat_table
        
    @staticmethod
    def chi2_pvalue(freq_table):
        from scipy.stats.contingency import chi2_contingency
        pvalue = chi2_contingency(freq_table).pvalue
        return round(pvalue, 3)
    
    def add_pchi2(self):
        assert self._cat_table.index.size==self.cat_table.index.size
        
        for v in self.catvars:
            pvalue=self.chi2_pvalue(self._cat_table.loc[v])
            # for indexing
            l1=v if self._cat_table.index.equals(self.cat_table.index) else self.varlab[v]
            l2=self.cat_table.loc[l1].index[0]
            self.cat_table.loc[(l1,l2), 'pvalue']=pvalue
        
        return self.cat_table
    
    def _fmt1(self):
        try:
            ALL=slice(None)
            n=self.cat_table.loc[:, (ALL, ALL, 'n')]
            p=self.cat_table.loc[:, (ALL, ALL, '%')]
            change_n=lambda x: f'{x:.0f} '
            change_p=lambda x: f'({x})'
            
            for i in range(n.shape[1]):
                vchange=n.iloc[:, i].apply(change_n)+p.iloc[:, i].apply(change_p)
                col_old=list(n.columns[i])
                col_old[2]='n (%)'
                col_new=tuple(col_old)
                self.cat_table.loc[:, col_new]=vchange
                
        except:
            self.cat_table=pd.DataFrame(self.cat_table.apply(lambda r: f"{r['n']:.0f} ({r['%']})", axis=1))
            self.cat_table.columns=['n (%)']
        
        try:
            pvalue=self.cat_table.loc[:, ('pvalue', ALL, ALL)]
            self.cat_table=pd.concat(
                [self.cat_table.loc[:, (ALL, ALL, 'n (%)')], pvalue]
                , axis=1)
        except:
            self.cat_table=self.cat_table.loc[:, (ALL, ALL, 'n (%)')]
            
        return self.cat_table
    
    def _fmt_idx(self):
        cols=self.cat_table.columns
        t1=self.cat_table.reset_index() \
            .drop_duplicates(('level_0','',''))
        t1.loc[:, ('level_1','','')]=t1.loc[:, ('level_0','','')]
        
        t1.loc[:, cols]=np.nan
        t2=self.cat_table.reset_index()
        t2.loc[:, ('level_1', '', '')]=t2.loc[:, ('level_1', '', '')].apply(lambda x: f'    {x}')

        self.cat_table=pd.concat(
            [t1,t2]).sort_index().drop(('level_0', '', ''), axis=1)
        self.cat_table.rename({('level_1', '', ''):('', '', '')}, axis=1)
        
        return self.cat_table
