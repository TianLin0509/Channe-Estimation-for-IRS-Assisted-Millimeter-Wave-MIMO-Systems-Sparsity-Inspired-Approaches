function [ix1]=BOMP(y,D,d,res)
% residual
% norm(y-D*ix1,'fro')< res
% 

[L,N]=size(D);
M=N/d;
r0=y;
iD=zeros(L,N);
niD=D;
ix0=zeros(N,1);
ix1 = zeros(N,1);
maxiter = 40;
for in1=1:maxiter
    if norm(y-D*ix1,'fro')>= 1*res

        vilo=niD'*r0;
        vil2=zeros(M,1);
        for in2=1:M
            vil2(in2)=norm(vilo((in2*d-d+1):in2*d));
        end
        [~,il]=max(vil2);
        iD(:,(il*d-d+1):il*d)=D(:,(il*d-d+1):il*d);
        indset=find(iD(1,:)~=0);
        nziD=iD(:,indset);
        %niD(:,(il*d-d+1):il*d)=zeros(L,d);
        %iD=[iD,D(:,(il*d-d+1):il*d)];
        nzix1=nziD\y;
        ix1=zeros(N,1);
        ix1(indset)=nzix1;
        r1=y-iD*ix1;
        if norm(ix0-ix1)<1e-2
            break;
        else
            ix0=ix1;
            r0=r1;
        end   
        
    end
end