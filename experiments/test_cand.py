import datetime
import pandas as pd

from hawkes.models import *

df = pd.read_csv('/Users/PauKung/Downloads/cand_influence_tweet1.csv',dtype={'Tweet_ID': object})

# create fomatted data
# TxKxF eventxevent_type counts
F_types = {'Immigration':0, 'Worker':1, 'Abortion':2, 'Foreign':3,
           'Health':4, 'Education':5, 'Terrorism':3, 'Muslim':3,
           'Tax':6, 'Economy':6, 'Trade':6, 'Budget':6, 'Govt':6,
           'Misc':7, 'Poll':8}
cand_idmap = {'Clinton':0, 'Sanders':1, 'Carson':2, 'Huckabee':3,
             'Cruz':4, 'Paul':5, 'Fiorina':6, 'Rubio':7, 'Walker':8,
             'Christie':9, 'Kasich':10, 'Bush':11, 'Trump':12}
sdate = datetime.datetime(2015, 7, 6)
edate = datetime.datetime(2015, 12, 1)
T = (edate-sdate).days + 1
K = 13
F = 10
data = np.zeros((T, K, F))
for row in df.iterrows():
    idx, d = row
    t = map(int, d.Time.split('/'))
    tstep = (datetime.datetime(t[2], t[0], t[1]) - sdate).days
    try:
        data[tstep, cand_idmap[d.From], F_types[d.Topic]] += 1
    except:
        data[tstep, cand_idmap[d.From], F_types['Misc']] += 1
    data[tstep, cand_idmap[d.From], -1] += 1
# KxKxT interaction counts

model = LinearDiscreteHawkes(K=13, dt=1, dt_max=3, reg="L1")
model.set_data(data)
model.fit_bfgs()