import requests
import xmltodict
from matplotlib import pyplot as plt
import numpy as np
from sklearn.covariance import EmpiricalCovariance
import networkx

key = 'OfhbNdJry1lQ0NHUsHxg'
secret = 'QhmMDRwq7sljJakiLlxduDxY3ZGNFxzONTL178aowIk'

myuserid = 110430434

aut_dict = {}
page = 1

resp = requests.get('https://www.goodreads.com/review/list.xml', 
    params=  {'key':key, 
        'v': 2,
        'page': page,
        'id': myuserid})

data_dict = xmltodict.parse(resp.content)['GoodreadsResponse']
bools = 'review' in data_dict['reviews'].keys()

while bools == True:
    reviews_books = [r for r in data_dict['reviews']['review']]
    for i in reviews_books:
        j = list(i.items())[1]
        rating = list(i.items())[2][1]
        if rating == '0':
            pass
        else:
            pgs = i['book']['num_pages']
            if pgs == None:
                pgs = 0
            true_rating = i['body'].split('a ')[-1][0:-3]
            title = j[1]['title'].split('(')[0]
            nt = title.split('— ')
            if len(nt) == 2:
                title = nt[1]
            else:
                title = nt[0]
            nt = title.split('—')
            if len(nt) == 2:
                title = nt[1]
            else:
                title = nt[0]
            aut_dict[title] = (title, true_rating, i['book']['average_rating'], pgs)
    page += 1
    resp = requests.get('https://www.goodreads.com/review/list.xml', 
        params=  {'key':key, 
            'v': 2,
            'page': page,
            'id': myuserid})
    data_dict = xmltodict.parse(resp.content)['GoodreadsResponse']
    bools = 'review' in data_dict['reviews'].keys()

z = list(aut_dict.values())
z = np.array(z)
zx = (z[:,2].astype(float))
pgs = (z[:,3].astype(float))
x = (z[:,1].astype(float))
dx = np.abs(x - zx)

bins = np.array([i/2 for i in range(0,12,1)])

dx_count,_ = np.histogram(dx,bins)
x_count,_ = np.histogram(x,bins)

if np.max(x_count) > np.max(dx_count):
    t_count = np.max(x_count)
else:
    t_count = np.max(dx_count)

fig, (ax, xa) = plt.subplots(2, 1,figsize=(7.5,10))
fig.subplots_adjust(top=0.9)
fig.suptitle('distribution of my literature ratings',fontsize = 20, fontname = 'Arial')

fig.text(0.06, 0.5, r'frequency, $f$', ha='center', va='center', rotation='vertical',fontsize = 18)

ax.set_xlabel(r'ratings, $r$',fontsize = 14, fontname = 'Arial')

ax.set_ylim((0,t_count + int(0.05*t_count)))
ax.set_yticks(np.arange(0, t_count, step=5))

xa.set_ylim((0,t_count + int(0.05*t_count)))
xa.set_yticks(np.arange(0, t_count, step=5))

xa.set_xlabel(r'ratings differential, $|r - \bar{r}|$',fontsize = 14, fontname = 'Arial')

xa.hist(dx,bins = [i/2 for i in range(0,12,1)],rwidth = 1,color = '#922724',edgecolor = 'black',align = 'left')
ax.hist(x,bins = [i/2 for i in range(0,12,1)],rwidth = 1,color = '#b1dbd0',edgecolor = 'black',align = 'left')

fig.tight_layout(pad=5.0)
xa.locator_params(axis='y', nbins=7)

for tick in ax.get_xticklabels():
    tick.set_fontname("Arial")
for tick in ax.get_yticklabels():
    tick.set_fontname("Arial")
plt.show()

A = EmpiricalCovariance().fit(np.array((x,dx))).covariance_
A = (A > 0) * A

A = A - np.diag(np.diag(A))
X = networkx.from_numpy_array(A)

F = networkx.Graph()
ps = networkx.spring_layout(X,scale=5, k = 1/len(A)**(1/40000))

labels = (z[:,0].astype(str))
l = {i: labels[i] for i in range(len(labels))}

networkx.draw_networkx_nodes(F,ps,nodelist = X.nodes,node_color = 'maroon',alpha = 0.35,node_size = 100)
networkx.draw_networkx_edges(F,ps,edgelist = X.edges,alpha = 0.05,edge_color = 'grey')
networkx.draw_networkx_labels(X,ps,l,font_size=6)

plt.axis('off')
plt.show()
