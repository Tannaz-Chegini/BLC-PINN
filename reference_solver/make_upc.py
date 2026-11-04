import numpy as np, time, sys
from fd_solver_upc import shishkin, solve
prob,eps,N=sys.argv[1],float(sys.argv[2]),int(sys.argv[3])
t0=time.time()
if prob=='P2':
    x=shishkin(N,2.5*eps*np.log(N)/0.5,(False,True)); y=shishkin(N,2.5*eps*np.log(N),(False,True))
    S=solve(eps,lambda X,Y:(1-X*Y/2,1+X*Y/2),lambda X,Y:0*X,lambda u:u**2,lambda X,Y,t:t**2*(X*(1-X)+Y*(1-Y)),2.0,x,y,int(0.8*N*2),store_times=(0.5,1.0,2.0))
else:
    x=shishkin(N,2.5*eps*np.log(N),(True,True)); y=x
    S=solve(eps**2,lambda X,Y:(0*X,0*Y),lambda X,Y:1+3*X+Y,lambda u:0*u,lambda X,Y,t:(1-np.exp(-5*t))*(1+X*Y),1.0,x,y,int(0.8*N),store_times=(0.25,0.5,1.0))
np.savez_compressed(f"upc_{prob}_eps{eps:g}_N{N}.npz",x=x,y=y,times=np.array(sorted(S)),**{f"u_{k:g}":v for k,v in S.items()})
tf=max(S); print(f"{prob} eps={eps:g} N={N}: max u {S[tf].max():.4f} {time.time()-t0:.0f}s",flush=True)
