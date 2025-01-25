#include <iostream>
#include <functional>
#include <cmath>
#include <Eigen/Dense> 
#include <Eigen/Sparse>
#include <fstream>
#include <cmath>
#include <cstdlib>




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
    
    for (int i = 0; i < 5; i++) {
        integral += weights[i] * f(rescale(nodes[i],a,b));
    }
    
    return integral*(b-a)/2;  
}




double e_i(int i, double x){
    if( i>n || x>2)
        return 0;

    if (( h*(i-1) <= x ) && (x <= h*i))
       return x/h - i + 1;
    else if (( h*(i) <= x ) && (x <= h*(i+1)))
       return -x/h + i + 1;
    else
       return 0;
}



double e_i_d(int i, double x){
    if( i>n || x>2)
        return 0;

    if (( h*(i-1) <= x ) && (x <= h*i))
       return 1/h ;
    else if (( h*(i) <= x ) && (x <= h*(i+1)))
       return -1/h ;
    else
       return 0;
}

double B_i_j(int i ,int j){
    // symetric matrix
    if (j < i) {
        std::swap(i, j);
    }

    double result = -1 * e_i( i, 2) * e_i( j, 2); // -w(2)v(2)

    if (j == i) {
        // diagonal
        auto f1 = [i](double x) {
            return e_i_d( i, x) * e_i_d( i, x); // w'v'
        };
        auto f2 = [i](double x) {
            return e_i( i, x) * e_i( i, x); // wv
        };
        
        result += integral(f1, (i - 1) * h, i * h) +
                  integral(f1, i * h, (i + 1) * h) -
                  integral(f2, (i - 1) * h, i * h) -
                  integral(f2, i * h, (i + 1) * h);

    } else if (j == i + 1) {
        // upper/lower diagonal
        auto f3 = [i,j](double x) {
            return e_i_d( i, x) * e_i_d( j, x); // w'v'
        };
        auto f4 = [i,j](double x) {
            return e_i( i, x) * e_i( j, x); // wv
        };
        
        result += integral(f3, (i - 1) * h, i * h) +
                  integral(f3, i * h, j * h) +
                  integral(f3, j * h, (j + 1) * h) -
                  integral(f4, (i - 1) * h, i * h) -
                  integral(f4, i * h, j * h) -
                  integral(f4, j * h, (j + 1) * h);
    } else {
        result = 0;
    }

    return result;
}

double L_j(int i){
    
    double result = e_i(0,2) * e_i( i, 2) + 4 * e_i( i, 2);

    auto f1 = [i](double x) {
        return e_i_d(0,x) * e_i_d( i, x); // u_tilde'(x) * v'(x)
    };

    auto f2 = [i](double x) {
        return e_i(i, x) * std::sin(x); // v(x) * sin(x)
    };

    auto f3 = [i](double x) {
        return e_i(0,x) * e_i( i, x); // u_tilde(x) * v(x)
    };

    result -= integral(f1, (i-1)*h, i*h);
    result -= integral(f1, i*h,(i+1)*h);

    result += integral(f2, (i-1)*h, i*h);
    result += integral(f2, i*h, (i+1)*h);

    result += integral(f3, (i-1)*h, i*h);
    result += integral(f3, i*h, (i+1)*h);

    return result;
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

    for(int j=1;j<=n;j++){
        for(int i=1;i<=n;i++){
           if(abs(i-j)<=1){
                triplets.push_back(Triplet<double>(j-1,i-1,B_i_j(i,j)));
           }
        }
        B(j-1)=L_j(j);
    }

 

    A.setFromTriplets(triplets.begin(),triplets.end());

    SparseLU<SparseMatrix<double>> solver;

    solver.compute(A);
    if(solver.info()!=Success){
        cout<<"Error A"<<endl;
        return 1;
    }

    VectorXd X = solver.solve(B);
    if(solver.info()!=Success){
        cout<<"Error B"<<endl;
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


    int result = system("python ./plot.py");
    if (result == 0) {
        cout << "Plot has been genarated" << endl;
        return 0;
    } else {
        cout << "Error while generating plot" << endl;
        return 1;
    }






  
   
    return 0;


    

  

  


 
}