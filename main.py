import numpy as np
import kmeans
import common
import naive_em
import em
##k-means algorithm
X = np.loadtxt("toy_data.txt")
mixture,post=common.init(X,3,0)
# k=[1,2,3,4] #No.Clusters
# rs=[0,1,2,3,4,5] #random states
# def mincost(X,k,rs):
#  for j in k:
#      previous_cost=None
#      for r in rs:
#          mixture,post=common.init(X,j,r)
#          mixture,post,cost=kmeans.run(X,mixture,post)
#          if previous_cost==None:
#             previous_cost=cost
#          elif cost< previous_cost:
#             previous_cost=cost   
#      print(previous_cost,j)      

# mincost(X,k,rs)     
#------------------------------------------------------#
u,o=naive_em.estep(X,mixture)
print(o)
       
        
        
        
       



