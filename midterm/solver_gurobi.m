function solver_gurobi()
clear all 
% graph size=n*n
n=64;
% solution size=num*num
num=n*n
Dist=[];
for i=1:num
    for j=1:num
        i_x=mod(i,n);
        j_x=mod(j,n);
        if i_x==0
            i_x=n;
        end
        if j_x==0
            j_x=n;
        end
        Dist(i,j)=sqrt((i_x-j_x)^2+(ceil(i./n)-ceil(j./n))^2);
    end
end
Dist=Dist+1.1*sparse(eye(num));
P=rand(num,num);

Cost=reshape(Dist,1,num*num);
b=P*ones(num,1); 
c=reshape(ones(1,num)*P,num,1);
R1=kron(sparse(eye(num)),sparse(ones(1,num)));
R2=kron(sparse(ones(1,num)),sparse(eye(num)));
PCost=Cost*reshape(P,num*num,1)
model.A = sparse([R1;R2]);
model.obj = Cost;
model.modelsense = 'Min';
model.rhs = [b;c];
model.sense = ['='];
%method option£º0 (Primal Simplex) 1(Dual Simplex) 2(Barrier Method/Interior Point Method)
params.Method=0

result = gurobi(model,params);

disp(result.objval);
reshape(result.x,num,num);
end