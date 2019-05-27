%#Knowledge Base.

cause(ldd, lnfs).
cause(lvd, lnfs).

%#impact(LDD, LVD, LNFS):- cause(LDD, LNFS); cause(LVD, LNFS).
impact(ldd, _, X):- cause(ldd, X).
impact(_, lvd, X):- cause(lvd, X).
