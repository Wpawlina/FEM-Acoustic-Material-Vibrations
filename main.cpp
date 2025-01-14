#include <iostream>
#include <functional>
#include <cmath>
#include <Eigen/Dense> 
#include <Eigen/Sparse>
#include <fstream>




using namespace std;
using namespace Eigen;


const int L = 2;
int n=0;
double h = 0;


double rescale(double x, double a, double b) {
    return ((b - a) * x + (b + a)) / 2.0;
}

double integral(function<double(double)>f,double a, double b){
    double nodes[5] = {0, 0.538469, -0.538469, 0.90618, -0.90618};
    double weights[5] = {0.568889, 0.478629, 0.478629, 0.236927, 0.236927};
    double integral = 0;
    
    for (int i = 0; i < 2; i++) {
        integral += weights[i] * f(rescale(nodes[i], a, b));
    }
    
    return integral;  
}




double e_i(int i, double x){
    if (( h*(i-1) <= x ) && (x < h*i))
       return x/h - i + 1;
    else if (( h*(i) <= x ) && (x <= h*(i+1)))
       return -x/h + i + 1;
    else
       return 0;
}



double e_i_d(int i, double x){
    if (( h*(i-1) <= x ) && (x < h*i))
       return 1/h ;
    else if (( h*(i) <= x ) && (x <= h*(i+1)))
       return -1/h ;
    else
       return 0;
}

double B_i_j(int i ,int j){
    double a=0;
    double b=0;
    if( i == j-1)
    {
        a=i*h;
        b=(i+1)*h;
    }
    else if(i == j+1)
    {
        a=(i-1)*h;
        b=i*h;
    }
    else if(i ==n )
    {
        a=(i-1)*h;
        b=i*h;
    }
    else
    {
        a=(i-1)*h;
        b=(i+1)*h;
    }

  return (-1)*e_i(i,2)*e_i(j,2)+integral([i,j]( double x){return e_i_d(i,x) *e_i_d(j,x);},a,b)-integral([i,j](double x){ return e_i(i,x)*e_i(j,x);},a,b);
}

double L_j(int j){
    
    auto sin1=[](double x){return sin(x);};
    double a=(j-1)*h;
    double b=(j+1)*h;

    return integral(sin1,a,b)-integral([j](double x){return e_i_d(j,x)*e_i_d(0,x);},a,b)+integral([j](double x){return e_i(j,x)*e_i(0,x);},a,b)+4*e_i(j,2);
}



int main(){
    cout << "Podaj n: ";
    cin >> n;
  
    if(n<1){
        cout<<"n musi byc wieksze od 0"<<endl;
        return 0;
    }

    h=(double)L/ (double)n;


    SparseMatrix<double> A(n,n);
    VectorXd B(n);

    vector<Triplet<double>> triplets;

    for(int j=0;j<n;j++){
        for(int i=0;i<n;i++){
            double value=B_i_j(i+1,j+1);
            if(value!=0.0){
                triplets.emplace_back(j,i,value);
            }
        }
        B(j)=L_j(j+1);
    }

    A.setFromTriplets(triplets.begin(),triplets.end());

    SparseLU<SparseMatrix<double>> solver;

    solver.compute(A);
    if(solver.info()!=Success){
        cout<<"Error"<<endl;
        return 1;
    }

    VectorXd X = solver.solve(B);
    if(solver.info()!=Success){
        cout<<"Error"<<endl;
        return 1;
    }


    VectorXd X_u(n+1);


  
     
    X_u(0)=0;
    for(int i=0;i<n;i++){
        X_u(i+1)=X(i);
    }
    for(int i=0;i<n+1;i++){
        X_u(i)=X_u(i)+e_i(0,i*h);
    }

    double x[n+1];
    for(int i=0;i<n+1;i++){
        x[i]=i*h;
    }
    double y[n+1];
    for(int i=0;i<n+1;i++){
        y[i]=X_u(i);
    }
    
    fstream file;

    file.open("data.csv",ios::out);
    for(int i=0;i<n+1;i++){
        file<<x[i]<<";"<<y[i]<<endl;
       
    }

    file.close(); 






  
   
    return 0;


    

  

  


 
}