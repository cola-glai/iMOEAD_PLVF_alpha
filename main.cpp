/*
 * Created by GuiyuLai on 2020/11/11.
 *
 * Institution:
 *  Computational Optimization for Learning and Adaptive System (COLA) Laboratory @ UESTC
 *
 * Copyright (c) 2020 GuiyuLai
 */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <unistd.h>
#include <cstring>
#include "eigen-3.3.8/Eigen/SVD"

using namespace std;

/*
 * fixed parameters
 */
const double EPS = 1.0e-7;
#define INF 0x3f3f3f3f
#define PI 3.1415926

/*
 * global variable
 */
double *z;//reference point z*
double **lambda;//weight vector metrix
int weight_vector_num;//number of weight vector
int **neighborhood;//B(i) for i=0,1,...,weight_vector_num-1
int index_0,index_1,index_2;
int flag_P;

/*
 * control parameters
 */
#define TEST_PROBLEM DTLZ1
#define K 5//parameter for more than 3-objective test problems
#define OBJECTIVE_NUM 5
#define VARIABLE_NUM (OBJECTIVE_NUM+K-1)
//#define VARIABLE_NUM 10
/*********
 * DTLZ1 [ITERATIVE_TIME 400] [Golden Point(0.2,0.15,0.15)]
 * DTLZ2 [ITERATIVE_TIME 250] [Golden Point(0.686,0.514,0.514)]
 * DTLZ3 [ITERATIVE_TIME 1000] [Golden Point(0.686,0.514,0.514)]
 * DTLZ4 [ITERATIVE_TIME 600] [Golden Point(0.686,0.514,0.514)]
 *
 * utopia weight (0.4,0.3,0.3)
 ********/
#define POPULATION_SIZE 210
#define ITERATIVE_TIME 3500
const double eta_m = 20;
const double MUTATION_PROBABILITY = 1.0/VARIABLE_NUM;
const double alpha=100;//parameter for DTLZ4
const int T=20;//number of weight vectors in the neighborhood of each weight vector
const double delta=0.9;//the probability that parent solutions are selected from the neighborhood otherwise, from the whole population
const int n_r=2;//the maximal number of solutions replaced by each child solution
const double CR=0.5;
const double F=0.5;

const int number_of_candidates=10;// for DM to score after 1st consultation
const int generations_between_consultations=15;
const double step_size=0.05;
int solutions_index_for_DM_to_score[2*OBJECTIVE_NUM+1];//C99之前不允许变量数组长度 number_of_candidates:=10
int index_x_best;
int index_weight_best;
/*weights that prefer the middle of the PF*/
//double w[3]={0.4,0.3,0.3};//utopia weight vector for 3 objectives
double w[5]={0.2,0.18,0.24,0.18,0.2};//utopia weight vector for 5 objectives
//double w[8]={0.13,0.13,0.12,0.13,0.14,0.12,0.11,0.11};//utopia weight vector for 8 objectives
//double w[10]={0.11,0.12,0.11,0.08,0.10,0.09,0.1,0.09,0.11,0.09};//utopia weight vector for 10 objectives

/*
 * define2string
 */
#define STRING1(R) #R
#define STRING2(R) STRING1(R)


/*
 * control saving format
 */
#define FOPEN_FILENAME "D:\\SoftwareInstallation\\JetBrains\\CLion\\WorkSpace\\iMOEAD_PLVF_DTLZ1_release\\FV\\" STRING2(TEST_PROBLEM)".txt"
#define FOPEN_MODE "w"

/*
 * struct
 */
struct SORT_LIST
{
    double euclidian_distance;
    int index;
};
struct INDIVIDUAL
{
    double x[VARIABLE_NUM];
    double f[OBJECTIVE_NUM];
    double score;//by DM
    int index;
    int flag;//能不能被选成邻居吸引  0-->可   1-->不可（promising weight 或者 已经被吸引过了）
};
/*boundaries for x_0,x_1,...,x_(VARIABLE_NUM-1)*/
double lower_boundary[VARIABLE_NUM];
double upper_boundary[VARIABLE_NUM];

/**pinv**/
// 利用Eigen库，采用SVD分解的方法求解矩阵伪逆，默认误差er为0
Eigen::MatrixXd pinv_eigen_based(Eigen::MatrixXd & origin, const float er = 0)
{
    // 进行svd分解
    Eigen::JacobiSVD<Eigen::MatrixXd> svd_holder(origin,
                                                 Eigen::ComputeThinU |
                                                 Eigen::ComputeThinV);
    // 构建SVD分解结果
    Eigen::MatrixXd U = svd_holder.matrixU();
    Eigen::MatrixXd V = svd_holder.matrixV();
    Eigen::MatrixXd D = svd_holder.singularValues();

    // 构建S矩阵
    Eigen::MatrixXd S(V.cols(), U.cols());
    S.setZero();

    for (unsigned int i = 0; i < D.size(); ++i) {

        if (D(i, 0) > er) {
            S(i, i) = 1 / D(i, 0);
        }
        else {
            S(i, i) = 0;
        }
    }

    // pinv_matrix = V * S * U^T
    return V * S * U.transpose();
}


/*
 * predefined function
 */
/*initialize weight*/
void set_weight (double *weight, double unit, double sum, int dim, int *column, double **lambda);
/*C(n,k)@select k from n*/
int combination (int n, int k);
/*num is the size of population|number_weight is pop_size*/
double **initialize_uniform_point (int num, int *weight_vector_num);
double euclidian_distance_between_two_vectors(double *a,double *b);
void sort_by_euclidian_distance(struct SORT_LIST *sort_list);
void neighborhood_of_vector_i(int i);
void neighborhood_of_all_vectors();
void initialize_boundary();
double rand_in_boundary(double lower_boundary,double upper_boundary);
void initialize_individual(struct INDIVIDUAL *individual,int i);
void initialize_population(struct INDIVIDUAL *individual);
/*test problems*/
void ZDT1(struct INDIVIDUAL *individual,int size);
void ZDT2(struct INDIVIDUAL *individual,int size);
void ZDT3(struct INDIVIDUAL *individual,int size);
void ZDT4(struct INDIVIDUAL *individual,int size);
void ZDT6(struct INDIVIDUAL *individual,int size);
void DTLZ1(struct INDIVIDUAL *individual,int size);
void DTLZ2(struct INDIVIDUAL *individual,int size);
void DTLZ3(struct INDIVIDUAL *individual,int size);
void DTLZ4(struct INDIVIDUAL *individual,int size);
void DTLZ5(struct INDIVIDUAL *individual,int size);
void DTLZ6(struct INDIVIDUAL *individual,int size);
void DTLZ7(struct INDIVIDUAL *individual,int size);
void initialize_ideal_point(struct INDIVIDUAL *individual);
void select_index_1_and_index_2(int i);
void differential_evolution(struct INDIVIDUAL *individual_0,struct INDIVIDUAL *individual_1,struct INDIVIDUAL *individual_2,struct INDIVIDUAL *y);
void update_ideal_point(struct INDIVIDUAL *y);
void mutation_on_y(struct INDIVIDUAL *y);
void compare_and_replace(struct INDIVIDUAL *y,struct INDIVIDUAL *x,int j,int *replace_num);
void random_permutation (int *perm, int size);
void update_of_solutions(int i,struct INDIVIDUAL *individual,struct INDIVIDUAL *y);
void update(struct INDIVIDUAL *individual,struct INDIVIDUAL *y);

/*MOEAD/D-PLVF*/
void solutions_for_DM_to_score_in_1st(struct INDIVIDUAL *individual);
void solutions_for_DM_to_score_after_1st(struct INDIVIDUAL *individual);//通过AVF打分，选最好(score最小)几个
void scoring_DM_1st(struct INDIVIDUAL *individual);//The 1st scoring via DM
void scoring_DM(struct INDIVIDUAL *individual);
void scoring_PLVF(struct INDIVIDUAL *individual);// via AVF
double Tchebycheff_Function(struct INDIVIDUAL *individual);
void neighbor_to_adjust(int i,int *neighbor,struct INDIVIDUAL *individual);
void update_weight_vector(struct INDIVIDUAL *individual);//double **lambda


void saving_FV(struct INDIVIDUAL *individual);
struct Q_START *readPF();



/****** RBF Net ******/
double eta = 0.8;//学习率
double ERR = 0.0005;//目标误差
const int ITERATION_CEIL = 500000;//最大训练次数

int SAMPLE_SIZE = 0;//输入样本的数量
double X[10][3];
double Y[10];
//中间层数量设为训练集大小
double center[10][3];
double sigma[10][3];//sigma
double weight[10];//权重矩阵
double error[10]={0};

double DM(double *x);
/*产生指定区间上均匀分布的随机数*/
double uniform(double floor, double ceil);
/*产生区间[floor,ceil]上服从正态分布N[mu,sigma]的随机数*/
double RandomNorm(double mu, double sigma, double floor, double ceil);
/*给向量赋予[floor,ceil]上的随机值*/
void Init_weight(double floor, double ceil);
void Init_center();
void Init_delta();
double getOutput(double * x);
/*计算单个样本引起的误差*/
double calSingleError(int index);
/*计算所有训练样本引起的总误差*/
double calTotalError();
/*更新网络参数*/
void updateParam();




int main()
{
    /*程序运行时间*/
    int begin,end;
    begin = time(NULL);
    srand(time(0));//使随机数种子随时间变化
    int metric_time=1;
    while (metric_time--)
    {
        /*initialize a uniform spread of weight vectors*/
        lambda=initialize_uniform_point(POPULATION_SIZE,&weight_vector_num);
        neighborhood_of_all_vectors();
        struct INDIVIDUAL individual[weight_vector_num];
        struct INDIVIDUAL *y=(struct INDIVIDUAL *)malloc(sizeof(struct INDIVIDUAL));
        initialize_population(individual);
        TEST_PROBLEM(individual,weight_vector_num);
        initialize_ideal_point(individual);
        int tao = generations_between_consultations;
        while (tao--)
        {
            update(individual,y);
        }
        printf("从以下population中选出7个解给DM打分\n");
        for (int i = 0; i < weight_vector_num; ++i)
        {
            for (int j = 0; j < OBJECTIVE_NUM; ++j)
            {
                printf("%f\t",individual[i].f[j]);
            }
            printf("\n");
        }
        solutions_for_DM_to_score_in_1st(individual);//2*OBJECTIVE_NUM+1个解
        scoring_DM_1st(individual);//第一次咨询同时将打分数据添加进训练集

        /*初始化网络参数*/
        Init_center();
        Init_delta();
//        Init_weight(0, 10);
        updateParam();//更新网络参数
//        printf("training set:\n");
//        for (int i = 0; i < SAMPLE_SIZE; ++i)
//        {
//            printf("input[%d]:%f %f %f\toutput[%d]:%f\n",i,X[i][0],X[i][1],X[i][2],i,Y[i]);
//        }

//        /*开始迭代*/
//        int iteration = ITERATION_CEIL;
//        double before,after;//记录一次迭代后总体均方误差
//        while (iteration-->0)
//        {
//            before = calTotalError();
////            printf("网络总体均方误差：%f\n",before);
//            if (before<ERR)//误差已达到要求，可以退出迭代||或者通过计算梯度向量的模，当模长小于一定阈值，说明收敛已达要求
//                break;
//            updateParam();//更新网络参数
//            after = calTotalError();
//            if(after>before)//减小学习率
//            {
//                eta = eta*0.9;
//                printf("步长更新为:%f\n",eta);
//                iteration = ITERATION_CEIL;//重新迭代
//                Init_weight(-5, 5);
//            }
//        }
//        printf("iteration times: %d\n",ITERATION_CEIL - iteration - 1);

        //用训练集测试下网络精度
//        for (int i = 0; i < SAMPLE_SIZE; ++i)
//        {
//            printf("真实:input[%d]:%f %f %f\toutput[%d]:%f\t",i,X[i][0],X[i][1],X[i][2],i,Y[i]);
//            printf("预测为:%f\n",getOutput(X[i]),Y[i]);
//        }
//        printf("_________________________________\n");
//        printf("|                               |\n");
//        printf("|                               |\n");
//        printf("|  Press Enter to continue...   |\n");
//        printf("|                               |\n");
//        printf("|                               |\n");
//        printf("|_______________________________|\n");
//        getchar();

        scoring_PLVF(individual);
        update_weight_vector(individual);
        neighborhood_of_all_vectors();
        int counter = generations_between_consultations + 1;
        while (counter<=ITERATIVE_TIME)
        {
            printf("\n----------------iteration %d----------------\n",counter);
            update(individual,y);
//            scoring_PLVF(individual);
            //saving_FV(individual);
            /*for (int i = 0; i < weight_vector_num; ++i)
            {
                printf("\nindividual[%d]:\t",i);
                for (int j = 0; j < OBJECTIVE_NUM; ++j)
                {
                    printf("f[%d]=%f ",j,individual[i].f[j]);
                }
                printf("index:=%d score:=%f\n",individual[i].index,individual[i].score);
            }*/
            if((counter-generations_between_consultations)%generations_between_consultations==0)//每隔一定的代数就让DM打分
            {
                printf("开始咨询...\n");
                scoring_PLVF(individual);
                solutions_for_DM_to_score_after_1st(individual);
                scoring_DM(individual);
                printf("现在是第 %d 次迭代，正在训练 RBF Net 中......\n",counter);
                /*动态初始化数据中心和拓展宽度*/
                Init_center();
                Init_delta();
                updateParam();//更新网络参数
//                Init_weight(-5, 5);
//                printf("training set:\n");
//                for (int i = 0; i < SAMPLE_SIZE; ++i)
//                {
//                    printf("input[%d]:%f %f %f\toutput[%d]:%f\n",i,X[i][0],X[i][1],X[i][2],i,Y[i]);
//                }
//                /*开始迭代*/
//                eta = 0.8;
//                iteration = ITERATION_CEIL;
//                while (iteration-->0)
//                {
//                    before = calTotalError();
//                    printf("网络总体均方误差：%f\n",before);
//                    if (before<ERR)//误差已达到要求，可以退出迭代||或者通过计算梯度向量的模，当模长小于一定阈值，说明收敛已达要求
//                        break;
//                    updateParam();//更新网络参数
//                    after = calTotalError();
//                    if(after>before)//减小学习率
//                    {
//                        eta = eta*0.96;
//                        printf("步长更新为:%f\n",eta);
//                        iteration = ITERATION_CEIL;//重新迭代
//                        Init_weight(-5, 5);
//                    }
//                }
////                ERR*=0.9;
//                printf("使用梯度下降法求解RBF Net参数的迭代次数 [iteration times]: %d\n",ITERATION_CEIL - iteration - 1);
                //用训练集测试下网络精度
//                for (int i = 0; i < SAMPLE_SIZE; ++i)
//                {
//                    printf("真实:input[%d]:%f %f %f\toutput[%d]:%f\t",i,X[i][0],X[i][1],X[i][2],i,Y[i]);
//                    printf("预测为:%f\n",getOutput(X[i]),Y[i]);
//                }
//                printf("第 %d 次迭代\n",counter);
//                printf("_________________________________\n");
//                printf("|                               |\n");
//                printf("|                               |\n");
//                printf("|  Press Enter to continue...   |\n");
//                printf("|                               |\n");
//                printf("|                               |\n");
//                printf("|_______________________________|\n");
//                getchar();
                scoring_PLVF(individual);
                update_weight_vector(individual);
                neighborhood_of_all_vectors();

            }
            counter++;
        }

//        /***
//         * 调试 RBFNet 试验场
//         ***/
//        //数据集归一化
//        double X_max,X_min;
//        //假设 min=0
//        for (int i = 0; i < OBJECTIVE_NUM; ++i)
//        {
//            X_min=X_max=X[0][i];
//            for (int j = 1; j < SAMPLE_SIZE; ++j)
//            {
//                if(X[j][i]>X_max)
//                {
//                    X_max=X[j][i];
//                }
//                else if(X[j][i]<X_min)
//                {
//                    X_min=X[j][i];
//                }
//            }
//            for (int j = 0; j < SAMPLE_SIZE; ++j)
//            {
//                X[j][i]=(X[j][i]-X_min)/(X_max-X_min);
//            }
//        }
//
//        double Y_max,Y_min;
//        Y_max=Y_min=Y[0];
//        for (int i = 1; i < SAMPLE_SIZE; ++i)
//        {
//            if(Y[i]>Y_max)
//            {
//                Y_max=Y[i];
//            }
//            else if (Y[i]<Y_min)
//            {
//                Y_min=Y[i];
//            }
//        }
//        for (int i = 0; i < SAMPLE_SIZE; ++i)
//        {
//            Y[i] = (Y[i])/(Y_max);
//        }
//
//        printf("############## training data ##############\n");
//        for (int i = 0; i < SAMPLE_SIZE; ++i)
//        {
//            printf("[%d] inputs: %f %f %f\toutputs: %f\n",i,X[i][0],X[i][1],X[i][2],Y[i]);
//        }
//        printf("[   按 Enter 键继续...   ]");
//        getchar();
//        Init_weight(-100, 100);
//        Init_center();
//        Init_delta();
//        printf("初始化的网络参数:\n");
//        for (int i = 0; i<SAMPLE_SIZE; ++i)
//        {
//            printf("第【%d】个节点参数%f %f %f %f %f %f %f\n",i,center[i][0],center[i][1],center[i][2],sigma[i][0],sigma[i][1],sigma[i][2],weight[i]);
//        }
//        /*开始迭代*/
//        int iteration = ITERATION_CEIL;
//        while (iteration-->0)
//        {
//            printf("网络总体均方误差：%f\n",calTotalError());
//            //误差已达到要求，可以退出迭代||或者通过计算梯度向量的模，当模长小于一定阈值，说明收敛已达要求
//            if (calTotalError()<ERR)
//            {
//                break;
//            }
//            updateParam();//更新网络权重参数
//        }
//        printf("iteration times: %d\n",ITERATION_CEIL - iteration - 1);
//        printf("训练后的此时网络的参数:");
//        for (int i = 0; i<SAMPLE_SIZE; ++i)
//        {
//            printf("第【%d】个节点参数%f %f %f %f %f %f %f\n",i,center[i][0],center[i][1],center[i][2],sigma[i][0],sigma[i][1],sigma[i][2],weight[i]);
//        }
//        //拿训练数据测试
//        double tempArray[OBJECTIVE_NUM];
//        for (int i = 0; i < SAMPLE_SIZE; ++i)
//        {
//            for (int j = 0; j < OBJECTIVE_NUM; ++j)
//            {
//                tempArray[j] = X[i][j]/X_max;
//            }
//            printf("\tprediction: %f ground-truth: %f\n",getOutput(tempArray),Y[i]/Y_max);
//        }
        //根据已训练好的神经网络作几组测试
        /*double tempArray[OBJECTIVE_NUM];
        printf("############## testing data ##############\n");
        for(int i=0;i<200;i++)
        {
            printf("testInput[%d]: ",i);
            for (int j = 0; j < OBJECTIVE_NUM; ++j)
            {
                tempArray[j] = uniform(0.1, 0.3)/X_max;
            }
            for (int k = 0; k < OBJECTIVE_NUM; ++k)
            {
                printf("%f\t",tempArray[k]);
            }
            printf("\tprediction: %f ground-truth: %f\n",getOutput(tempArray),DM(tempArray));
        }*/
        printf("lastest population:\n");
        for (int i = 0; i < weight_vector_num; ++i)
        {
            printf("individual[%d]:",i);
            for (int j = 0; j < OBJECTIVE_NUM; ++j)
            {
                printf("f[%d]=%f ",j,individual[i].f[j]);
            }
            printf("\n");
        }
        saving_FV(individual);
        /*struct Q_START *q_head=NULL;
        q_head=readPF();
        double IGD=calculate_IGD(individual,q_head);
        saving_IGD(IGD);
        double HV=calculate_HV(individual,weight_vector_num);
        saving_HV(HV);*/
        free(y);
        free(z);
        for (int i = 0; i < weight_vector_num; ++i)
        {
            printf("weight vector [%d]:",i);
            for (int j = 0; j < OBJECTIVE_NUM; ++j)
            {
                printf("%f ",lambda[i][j]);
            }
            printf("\n");
        }
        free(lambda);
        sleep(1);
        printf("remaining %d run\n",metric_time);
    }
    printf("SAMPLE_SIZE:= %d\n",SAMPLE_SIZE);
    end = time(NULL);
    printf("Progam's running time: %ds\n",end - begin);
    return 0;
}
//main end

void set_weight (double *weight, double unit, double sum, int dim, int *column, double **lambda)
{
    int i;

    if (dim == OBJECTIVE_NUM)
    {
        for ( i = 0; i < OBJECTIVE_NUM; i++)
            weight[i] = 0;
    }
    if (dim == 1)//possible|dim is just equal to objective_number in the beginning
    {
        weight[0] = unit - sum;
        for ( i = 0; i < OBJECTIVE_NUM; i++)
            lambda[*column][i] = weight[i];
        *column = *column + 1;
        return;
    }
    for (i = 0; i <= unit - sum; i++)
    {
        weight[dim - 1] = i;
        set_weight (weight, unit, sum + i, dim - 1, column, lambda);
    }
    return;
}
int combination (int n, int k)
{
    int i;

    if (n < k)
        return -1;
    double ans = 1;
    for (i = k + 1; i <= n; i++)
    {
        ans = ans * i;
        ans = ans / (double) (i - k);
    }

    return (int) ans;
}
/*num is the size of population|weight_vector_num is new_pop_size*/
double **initialize_uniform_point (int num, int *weight_vector_num)
{
    int i, j;

    int layer_size;
    int column = 0;

    double *Vec;
    double **lambda = NULL;

    int gaps = 1;

    *weight_vector_num = 0;
    while(1)
    {
        layer_size  = combination (OBJECTIVE_NUM + gaps - 1, gaps);
        //printf("[%d]%d\n",gaps,layer_size);

        if(layer_size > num) break;
        *weight_vector_num = layer_size;
        gaps = gaps + 1;

    }
    gaps = gaps - 1;
    lambda = (double **) malloc ((*weight_vector_num) * sizeof(double *));//number_weight is pop_size
    for (i = 0; i < *weight_vector_num; i++)
    {
        lambda[i] = (double *) malloc(OBJECTIVE_NUM  * sizeof(double));
    }

    Vec = (double *) malloc (OBJECTIVE_NUM  * sizeof(double));
    for (i = 0; i < OBJECTIVE_NUM ; i++)
        Vec[i] = 0;
    set_weight (Vec, gaps, 0, OBJECTIVE_NUM, &column, lambda);

    for (i = 0; i < *weight_vector_num; i++)
        for (j = 0; j < OBJECTIVE_NUM; j++) {
            lambda[i][j] = lambda[i][j] / gaps;
        }

    free (Vec);

    return lambda;
}
double euclidian_distance_between_two_vectors(double *a,double *b)
{
    double distance;
    distance=0.0;
    for (int i = 0; i < OBJECTIVE_NUM; ++i)
    {
        distance+=(a[i]-b[i])*(a[i]-b[i]);
    }
    return sqrt(distance);
}
void sort_by_euclidian_distance(struct SORT_LIST *sort_list)
{
    struct SORT_LIST temp;
    /*bubble only for T times*/
    for (int i = 0; i < T; ++i)
    {
        for (int j = weight_vector_num-1; j >  i; --j)
        {
            if (sort_list[j].euclidian_distance<sort_list[j-1].euclidian_distance)
            {
                temp=sort_list[j-1];
                sort_list[j-1]=sort_list[j];
                sort_list[j]=temp;
            }
        }
    }
}
void neighborhood_of_vector_i(int i)
{
    struct SORT_LIST sort_list[weight_vector_num];
    neighborhood[i]=(int *)malloc(T*sizeof(int));
    for (int j = 0; j < weight_vector_num; ++j)
    {
        sort_list[j].euclidian_distance=euclidian_distance_between_two_vectors(lambda[i],lambda[j]);//当前第i个权重向量与其余所有weight_vector_num个权重向量的距离
        sort_list[j].index=j;
    }
    sort_by_euclidian_distance(sort_list);//sort by euclidian_distance|just find the minimal T index
    for (int j = 0; j < T; ++j)
    {
        neighborhood[i][j]=sort_list[j].index;
    }
    return;
}
void neighborhood_of_all_vectors()
{
    neighborhood=(int **)malloc(weight_vector_num*sizeof(int *));
    for (int i = 0; i < weight_vector_num; ++i)
    {
        neighborhood_of_vector_i(i);
    }
    return;
}
void initialize_boundary()
{
    lower_boundary[0]=0.0;
    upper_boundary[0]=1.0;
    if(strcmp(STRING2(TEST_PROBLEM),"ZDT4")==0)
    {
        for (int i = 1; i < VARIABLE_NUM; ++i)
        {
            lower_boundary[i]=-5.0;
            upper_boundary[i]=5.0;
        }
    }
    else
    {
        for (int i = 1; i < VARIABLE_NUM; ++i)
        {
            lower_boundary[i]=0.0;
            upper_boundary[i]=1.0;
        }
    }
    return;
}
double rand_in_boundary(double lower_boundary,double upper_boundary)
{
    double rand_in_0_1;
    rand_in_0_1=(double)rand()/RAND_MAX;
    return lower_boundary+(upper_boundary-lower_boundary)*rand_in_0_1;
}
void initialize_individual(struct INDIVIDUAL *individual,int i)
{
    individual->index = i;
    individual->flag = 0;//可以当成即将被吸引的邻居
    for (int j = 0; j < VARIABLE_NUM; ++j)
    {
        individual->x[j]=rand_in_boundary(lower_boundary[j],upper_boundary[j]);
    }
}
void initialize_population(struct INDIVIDUAL *individual)
{
    srand((unsigned)time(NULL));
    initialize_boundary();
    for (int i = 0; i < weight_vector_num; ++i)
    {
        initialize_individual(individual+i,i);
    }
    return;
}
void ZDT1(struct INDIVIDUAL *individual,int size)
{
    double g,h,sum;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0;
        p->f[0]=p->x[0];
        for(int j=1;j<VARIABLE_NUM;j++)
        {
            sum+=p->x[j];//Pareto-optimal front, for all x[i]=0
        }
        g=1+(9*sum)/(VARIABLE_NUM-1);
        h=1-sqrt((double)(p->f[0]/g));
        p->f[1]=g*h;
    }
    return;
}
void ZDT2(struct INDIVIDUAL *individual,int size)
{
    double g,h,sum;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0;
        p->f[0]=p->x[0];
        for(int j=1;j<VARIABLE_NUM;j++)
        {
            sum+=p->x[j];
        }
        g=1+(9*sum)/(VARIABLE_NUM-1);
        h=1-pow(p->f[0]/g,2);
        p->f[1]=g*h;
    }
    return;
}
void ZDT3(struct INDIVIDUAL *individual,int size)
{
    double g,h,sum;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0;
        p->f[0]=p->x[0];
        for(int j=1;j<VARIABLE_NUM;j++)
        {
            sum+=p->x[j];
        }
        g=1+(9*sum)/(VARIABLE_NUM-1);
        h=1-sqrt((double)(p->f[0]/g))-p->f[0]*sin(10.0*PI*p->f[0])/g;
        p->f[1]=g*h;
    }
    return;
}
void ZDT4(struct INDIVIDUAL *individual,int size)
{
    double g,h,sum;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0;
        p->f[0]=p->x[0];
        for(int j=1;j<VARIABLE_NUM;j++)
        {
            sum+=pow(p->x[j],2.0)-10.0*cos(4.0*PI*p->x[j]);
        }
        g=1.0+10.0*(VARIABLE_NUM-1)+sum;
        h=1.0-sqrt((double)(p->f[0]/g));
        p->f[1]=g*h;
    }
    return;
}
void ZDT6(struct INDIVIDUAL *individual,int size)
{
    double g,h,sum;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0;
        p->f[0]=1-exp(-4*p->x[0])*pow(sin(6.0*PI*p->x[0]),6);
        for(int j=1;j<VARIABLE_NUM;j++)
        {
            sum+=p->x[j];
        }
        g=1+9*(pow(sum/9.0,0.25));
        h=1-pow(p->f[0]/g,2);
        p->f[1]=g*h;
    }
    return;
}
void DTLZ1(struct INDIVIDUAL *individual,int size)//todo: extend to any dimension
{
    int temp;
    double g,sum;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0.0;
        for(int j=OBJECTIVE_NUM-1;j<VARIABLE_NUM;j++)
        {
            sum+=pow((p->x[j]-0.5),2.0)-cos(20.0*PI*(p->x[j]-0.5));
        }
        g=100*(K+sum);//K=5 is suggested
        for (int j = 0; j < OBJECTIVE_NUM; ++j)
        {
            p->f[j] = 0.5*(1+g);
            temp = 0;
            while (temp<OBJECTIVE_NUM-j-1)
            {
                p->f[j]*=p->x[temp];
                temp++;
            }
            if (temp<OBJECTIVE_NUM-1)
            {
                p->f[j]*=(1-p->x[temp]);
            }
        }
    }
    return;
}
void DTLZ2(struct INDIVIDUAL *individual,int size)
{
    double g,sum,x1,x2;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0.0;
        x1=p->x[0];
        x2=p->x[1];
        for(int j=2;j<VARIABLE_NUM;j++)
        {
            sum+=pow((p->x[j]-0.5),2.0);
        }
        g=sum;//K=10
        p->f[0]=(1+g)*cos(x1*PI*0.5)*cos(x2*PI*0.5);//f1
        p->f[1]=(1+g)*cos(x1*PI*0.5)*sin(x2*PI*0.5);//f2
        p->f[2]=(1+g)*sin(x1*PI*0.5);//f3
    }
    return;
}
void DTLZ3(struct INDIVIDUAL *individual,int size)
{
    double g,sum,x1,x2;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0.0;
        x1=p->x[0];
        x2=p->x[1];
        for(int j=2;j<VARIABLE_NUM;j++)
        {
            sum+=pow((p->x[j]-0.5),2.0)-cos(20.0*PI*(p->x[j]-0.5));
        }
        g=100*(K+sum);//K=10
        p->f[0]=(1+g)*cos(x1*PI*0.5)*cos(x2*PI*0.5);//f1
        p->f[1]=(1+g)*cos(x1*PI*0.5)*sin(x2*PI*0.5);//f2
        p->f[2]=(1+g)*sin(x1*PI*0.5);//f3
    }
    return;
}
void DTLZ4(struct INDIVIDUAL *individual,int size)
{
    double g,sum,x1,x2;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0.0;
        x1=p->x[0];
        x2=p->x[1];
        for(int j=2;j<VARIABLE_NUM;j++)
        {
            sum+=pow((p->x[j]-0.5),2.0);
        }
        g=sum;//K=10
        p->f[0]=(1+g)*cos(pow(x1,alpha)*PI*0.5)*cos(pow(x2,alpha)*PI*0.5);//f1
        p->f[1]=(1+g)*cos(pow(x1,alpha)*PI*0.5)*sin(pow(x2,alpha)*PI*0.5);//f2
        p->f[2]=(1+g)*sin(pow(x1,alpha)*PI*0.5);//f3
    }
    return;
}
void DTLZ5(struct INDIVIDUAL *individual,int size)
{
    double sum,g,x1,x2,sita1,sita2;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0.0;
        x1=p->x[0];
        x2=p->x[1];
        for(int j=2;j<VARIABLE_NUM;j++)
        {
            sum+=pow((p->x[j]-0.5),2.0);
        }
        g=sum;//K=10
        sita1=PI*x1*0.5;//  BUG!!!->SOLVED
        sita2=PI*(1+2*g*x2)/(4*(1+g));
        p->f[0]=(1+g)*cos(sita1)*cos(sita2);//f1
        p->f[1]=(1+g)*cos(sita1)*sin(sita2);//f2
        p->f[2]=(1+g)*sin(sita1);//f3
    }
    return;
}
void DTLZ6(struct INDIVIDUAL *individual,int size)
{
    double sum,g,x1,x2,sita1,sita2;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0.0;
        x1=p->x[0];
        x2=p->x[1];
        for(int j=2;j<VARIABLE_NUM;j++)
        {
            sum+=pow(p->x[j],0.1);
        }
        g=sum;//K=10
        sita1=PI*x1*0.5;//  BUG!!!->SOLVED
        sita2=PI*(1+2*g*x2)/(4*(1+g));
        p->f[0]=(1+g)*cos(sita1)*cos(sita2);//f1
        p->f[1]=(1+g)*cos(sita1)*sin(sita2);//f2
        p->f[2]=(1+g)*sin(sita1);//f3
    }
    return;
}
void DTLZ7(struct INDIVIDUAL *individual,int size)
{
    double sum,g,h;
    struct INDIVIDUAL *p;
    for (int i = 0; i < size; ++i)
    {
        p=individual+i;
        sum=0.0;
        p->f[0]=p->x[0];//f1
        p->f[1]=p->x[1];//f2
        for(int j=2;j<VARIABLE_NUM;j++)
        {
            sum+=p->x[j];
        }
        g=1+0.45*sum;//K=10
        h=3-(p->f[0]*(1+sin(3*PI*p->f[0]))+p->f[1]*(1+sin(3*PI*p->f[1])))/(1+g);
        p->f[2]=g*h;//f3
    }
    return;
}
/*find the min on all objective in the current population*/
void initialize_ideal_point(struct INDIVIDUAL *individual)
{
    z=(double *)malloc(sizeof(double)*OBJECTIVE_NUM);
    struct INDIVIDUAL *p;
    /*memset(z,0x43,OBJECTIVE_NUM);*/
    for (int i = 0; i < OBJECTIVE_NUM; ++i) {
        z[i]=INF;
    }
    for (int i = 0; i < weight_vector_num; ++i)
    {
        p=individual+i;
        for (int j = 0; j < OBJECTIVE_NUM; ++j)
        {
            if (p->f[j]<z[j])
            {
                z[j]=p->f[j];
            }
        }
    }
    return;
}
void select_index_1_and_index_2(int i)
{
    double rand_in_0_1;
    int rand_1,rand_2;
    rand_in_0_1=(double)rand()/RAND_MAX;
    if (rand_in_0_1<delta)//select in B(I)
    {
        flag_P=0;
        rand_1=rand()%T;
        while (rand_1==(rand_2=rand()%T));
        index_0=i;//neighborhood[i][0];
        /*index_1 and index_2 are all global variable*/
        index_1=neighborhood[i][rand_1];
        index_2=neighborhood[i][rand_2];
    }
    else//select from whole population
    {
        flag_P=1;
        rand_1=rand()%weight_vector_num;
        while (rand_1==(rand_2=rand()%weight_vector_num));
        index_0=i;
        index_1=rand_1;
        index_2=rand_2;
    }
    return;
}
void differential_evolution(struct INDIVIDUAL *individual_0,struct INDIVIDUAL *individual_1,struct INDIVIDUAL *individual_2,struct INDIVIDUAL *y)
{
    double rand_in_0_1;
    int r=rand()%VARIABLE_NUM;//[0,1,...,VARIABLE_NUM-1]
    for (int i = 0; i < VARIABLE_NUM; ++i)
    {
        rand_in_0_1=(double)rand()/RAND_MAX;
        if (rand_in_0_1<CR||i==r)
        {
            y->x[i]=individual_2->x[i]+F*(individual_0->x[i]-individual_1->x[i]);
            y->x[i]=(y->x[i]<lower_boundary[i])?lower_boundary[i]:(y->x[i]>upper_boundary[i])?upper_boundary[i]:y->x[i];
        }
        else
        {
            y->x[i]=individual_2->x[i];
        }
    }
    return;
}
void mutation_on_y(struct INDIVIDUAL *y)
{
    double ri,deta_i;
    for (int i = 0; i < VARIABLE_NUM; ++i)
    {
        if((double)rand()/RAND_MAX>MUTATION_PROBABILITY+EPS)
        {
            continue;
        }
        /*printf("mutation occurs!\n");*/
        ri=(double)rand()/RAND_MAX;
        while (fabs(ri-1)<EPS)
        {
            ri=(double)rand()/RAND_MAX;
        }
        if(ri+EPS<0.5)//ri<0.5
        {
            deta_i=pow(2*ri,1.0/(eta_m+1))-1;
        }
        else//ri>=0.5
        {
            deta_i=1-pow(2*(1-ri),1.0/(eta_m+1));
        }
        y->x[i]+=deta_i*(upper_boundary[i]-lower_boundary[i]);
        if(y->x[i]<lower_boundary[i])
        {
            y->x[i]=lower_boundary[i];
        }
        else if(y->x[i]>upper_boundary[i])
        {
            y->x[i]=upper_boundary[i];
        }
    }
    return;
}
void update_ideal_point(struct INDIVIDUAL *y)
{
    for (int i = 0; i < OBJECTIVE_NUM; ++i)
    {
        if (z[i]>y->f[i])
        {
            z[i]=y->f[i];
        }
    }
}
void compare_and_replace(struct INDIVIDUAL *y,struct INDIVIDUAL *x,int j,int *replace_num)
{
    y->index = x->index;
    y->flag = 0;
    double g_y_max,g_x_max;
    g_y_max=g_x_max=-1.0e+30;
    for (int i = 0; i < OBJECTIVE_NUM; ++i)
    {
        if (lambda[j][i]<EPS)
        {
            if (fabs(y->f[i]-z[i])/0.000001>g_y_max)
            {
                g_y_max=fabs(y->f[i]-z[i])/0.000001;
            }
            if (fabs(x->f[i]-z[i])/0.000001>g_x_max)
            {
                g_x_max=fabs(x->f[i]-z[i])/0.000001;
            }
        }
        else
        {
            if (fabs(y->f[i]-z[i])/lambda[j][i]>g_y_max)
            {
                g_y_max=fabs(y->f[i]-z[i])/lambda[j][i];
            }
            if (fabs(x->f[i]-z[i])/lambda[j][i]>g_x_max)
            {
                g_x_max=fabs(x->f[i]-z[i])/lambda[j][i];
            }
        }
    }
    if (g_y_max+EPS<g_x_max)
    {
        *x=*y;
        (*replace_num)++;
    }
    return;
}
void random_permutation (int *perm, int size)
{
    int i, num, start;
    int *index, *flag;

    index = (int *)malloc (size * sizeof(int));
    flag  = (int *)malloc (size * sizeof(int));
    for (i = 0; i < size; i++)
    {
        index[i] = i;
        flag[i]  = 1;
    }

    num = 0;
    while (num < size)//perm is not full
    {
        start = rand()%size;
        while (1)
        {
            if (flag[start])
            {
                perm[num] = index[start];
                flag[start] = 0;
                num++;
                break;
            }
            if (start == (size - 1))
                start = 0;
            else
                start++;//next
        }
    }

    free (index);
    free (flag);
    return;
}
void update_of_solutions(int i,struct INDIVIDUAL *individual,struct INDIVIDUAL *y)
{
    /*printf("B(%d):",i);
    for (int j = 0; j < T; ++j)
    {
        printf("%d ",neighborhood[i][j]);
    }
    printf("\n");*/
    int index;
    int replace_num=0;
    if (flag_P==0)//replace some in B(i)
    {
        int perm[T];
        random_permutation(perm,T);
        /*for (int i = 0; i < T; ++i) {
            printf("perm[%d]=%d ",i,perm[i]);
        }
        printf("\n");*/
        for (int j = 0; j < T; ++j)
        {
            if(replace_num>=n_r)
            {
                break;
            }
            index=neighborhood[i][perm[j]];

            compare_and_replace(y,individual+index,index,&replace_num);
        }

    }
    else if (flag_P==1)//replace some in whole population
    {
        int perm[weight_vector_num];
        random_permutation(perm,weight_vector_num);
        /*for (int i = 0; i < T; ++i) {
            printf("perm[%d]=%d ",i,perm[i]);
        }
        printf("\n");*/
        for (int j = 0; j < weight_vector_num; ++j)
        {
            if(replace_num>=n_r)
            {
                break;
            }
            index=perm[j];
            compare_and_replace(y,individual+index,index,&replace_num);
        }
    }
    return;
}
void update(struct INDIVIDUAL *individual,struct INDIVIDUAL *y)
{
    for (int i = 0; i < weight_vector_num; ++i)
    {
        select_index_1_and_index_2(i);
        /*printf("flag_P=%d index_0=%d index_1=%d index_2=%d\n",flag_P,index_0,index_1,index_2);*/
        differential_evolution(individual+index_0,individual+index_1,individual+index_2,y);
        /*printf("individual[%d]'s offspring by DE:",i);
        for (int j = 0; j < VARIABLE_NUM; ++j)
        {
            printf("x[%d]=%f ",j,y->x[j]);
        }
        for (int j = 0; j < OBJECTIVE_NUM; ++j)
        {
            printf("f[%d]=%f ",j,y->f[j]);
        }
        printf("\n");*/
        mutation_on_y(y);
        TEST_PROBLEM(y,1);//calculate FV for the new offspring--y
        update_ideal_point(y);
        /*printf("update solutions around %d: \n",i);*/
        update_of_solutions(i,individual,y);
    }
    return;
}

/*only search for the top mu solution
the corresponding weight vector is sighed by index*/
void solutions_for_DM_to_score_in_1st(struct INDIVIDUAL *individual)//todo:空间中均匀选点来打分
{
    int mu = 2*OBJECTIVE_NUM + 1;
    double **lambda_true;
    int num=POPULATION_SIZE;//todo:to control how many weight vectors
    int mu_true;
    lambda_true=initialize_uniform_point(num,&mu_true);
    printf("%d\n",mu_true);
    for (int i = 0; i < mu_true; ++i)
    {
        printf("Zone_lambda[%d]:(\t",i);
        for (int j = 0; j < OBJECTIVE_NUM; ++j)
        {
            printf("%f\t",lambda_true[i][j]);
        }
        printf(")\n");
    }
    for (int i = 0; i < mu; ++i)
    {
        solutions_index_for_DM_to_score[i] = rand()%weight_vector_num;
    }
    return;
}

void solutions_for_DM_to_score_after_1st(struct INDIVIDUAL *individual)
{
    int u[number_of_candidates];
    struct INDIVIDUAL array[weight_vector_num];
    for (int j = 0; j < weight_vector_num; ++j)
    {
        array[j] = individual[j];
    }
    struct INDIVIDUAL temp;
    for (int i = 0; i < number_of_candidates; ++i)
    {
        for (int j = weight_vector_num-1; j > i; --j)
        {
            if (array[j].score<array[j-1].score)
            {
                temp = array[j-1];
                array[j-1] = array[j];
                array[j] = temp;
            }
        }
    }
    printf("选出给DM打分的解下标:\t");
    int cnt=0;
    for (int k = 0; k <weight_vector_num ; k++)
    {
        bool flag = false;
        for (int i = 0; i < OBJECTIVE_NUM; ++i)
        {
            if (fabs(array[k].f[i]-array[k+1].f[i])>1e-6)
            {
                flag = true;
                break;
            }
        }
        if (flag)
        {
            printf("%d ",array[k].index);
            u[cnt] = array[k].index;
            cnt++;
            if (cnt>=10)
            {
                break;
            }
        }
    }
    if (cnt<10)
    {
        printf("error: please choose more candidates");
        exit(1);
    }
    printf("\n");
    for (int i = 0; i < number_of_candidates; ++i)
    {
        solutions_index_for_DM_to_score[i] = u[i];
    }
    return;
}

void scoring_DM_1st(struct INDIVIDUAL *individual)
{
    int mu = 2*OBJECTIVE_NUM + 1;
    double best_score = INF;//找出最佳（最小）评分
    for (int i = 0; i < mu; ++i)
    {
        //scanf("%lf",&(individual[solutions_index_for_DM_to_score[i]].score));//手动评太慢
        double a[OBJECTIVE_NUM];
        double Tchebycheff = EPS;
        for (int j = 0; j < OBJECTIVE_NUM; ++j)
        {
            a[j] = individual[solutions_index_for_DM_to_score[i]].f[j];
            if (fabs(a[j]-z[j])/w[j]>Tchebycheff)
            {
                Tchebycheff = fabs(a[j]-z[j])/w[j];
            }
        }
        individual[solutions_index_for_DM_to_score[i]].score = Tchebycheff;

        printf("DM对个体[%d]\t(",individual[solutions_index_for_DM_to_score[i]]);
        //样本加入训练集
        for (int j = 0; j <OBJECTIVE_NUM; ++j)
        {
            printf("%f\t",individual[solutions_index_for_DM_to_score[i]].f[j]);
            X[SAMPLE_SIZE][j] = individual[solutions_index_for_DM_to_score[i]].f[j];
        }
        printf(")打分：%f\n",Tchebycheff);
        //Y[SAMPLE_SIZE] = individual[solutions_index_for_DM_to_score[i]].score + RandomNorm(0, 0.1, -0.1, 0.1);
        Y[SAMPLE_SIZE] = individual[solutions_index_for_DM_to_score[i]].score;
        SAMPLE_SIZE++;
        if (individual[solutions_index_for_DM_to_score[i]].score < best_score)
        {
            best_score = individual[solutions_index_for_DM_to_score[i]].score;
            /*solutions_index_for_DM_to_score[i]==individual[solutions_index_for_DM_to_score[i]].index:=1*/
            index_x_best = individual[solutions_index_for_DM_to_score[i]].index;
            index_weight_best = individual[solutions_index_for_DM_to_score[i]].index;
        }
    }
    return;
}
void scoring_DM(struct INDIVIDUAL *individual)
{
    SAMPLE_SIZE = 0;
    int mu = number_of_candidates;
    double best_score = INF;//找出最佳（最小）评分
    for (int i = 0; i < mu; ++i)
    {
        //scanf("%lf",&(individual[solutions_index_for_DM_to_score[i]].score));
        double a[OBJECTIVE_NUM];
        double Tchebycheff = EPS;
        for (int j = 0; j < OBJECTIVE_NUM; ++j)
        {
            a[j] = individual[solutions_index_for_DM_to_score[i]].f[j];
            if (fabs(a[j]-z[j])/w[j]>Tchebycheff)
            {
                Tchebycheff = fabs(a[j]-z[j])/w[j];
            }
        }
        individual[solutions_index_for_DM_to_score[i]].score = Tchebycheff;

        printf("DM对个体[%d]\t(",individual[solutions_index_for_DM_to_score[i]]);
        //样本加入训练集
        for (int j = 0; j <OBJECTIVE_NUM; ++j)
        {
            printf("%f\t",individual[solutions_index_for_DM_to_score[i]].f[j]);
            X[SAMPLE_SIZE][j] = individual[solutions_index_for_DM_to_score[i]].f[j];
        }
        printf(")打分：%f\n",Tchebycheff);
        //Y[SAMPLE_SIZE] = individual[solutions_index_for_DM_to_score[i]].score + RandomNorm(0, 0.1, -0.1, 0.1);
        Y[SAMPLE_SIZE] = individual[solutions_index_for_DM_to_score[i]].score;
        SAMPLE_SIZE++;
        if (individual[solutions_index_for_DM_to_score[i]].score < best_score)
        {
            best_score = individual[solutions_index_for_DM_to_score[i]].score;
            /*solutions_index_for_DM_to_score[i]==individual[solutions_index_for_DM_to_score[i]].index:=1*/
            index_x_best = individual[solutions_index_for_DM_to_score[i]].index;
            index_weight_best = individual[solutions_index_for_DM_to_score[i]].index;
        }
    }
    return;
}
void scoring_PLVF(struct INDIVIDUAL *individual)
{
    printf("##############RBFNN对所有个体打分##############\n");
    for (int j = 0; j < weight_vector_num; ++j)
    {
        individual[j].score = 0.0;
        for (int i = 0; i<SAMPLE_SIZE; ++i)//控制基函数中心
        {
            individual[j].score += weight[i] * exp(-1.0*sqrt((individual[j].f[0] - center[i][0])*(individual[j].f[0] - center[i][0]) / (2 * sigma[i][0] * sigma[i][0]) + (individual[j].f[1] - center[i][1])*(individual[j].f[1] - center[i][1]) / (2 * sigma[i][1] * sigma[i][1]) + (individual[j].f[2] - center[i][2])*(individual[j].f[2] - center[i][2]) / (2 * sigma[i][2] * sigma[i][2])));
        }

        printf("individual[%d](\t",j);
        for (int i = 0; i < OBJECTIVE_NUM; ++i)
        {
            printf("%f\t",individual[j].f[i]);
        }
        printf(")自动评分:%f\n",individual[j].score);
    }
    return;
}
/*按照得分将population排序
从小到大排,仅排出前number_of_candidates个solutions用来调整权重就好*/
double Tchebycheff_Function(struct INDIVIDUAL *individual)
{
    double g = EPS;
    for (int j = 0; j < OBJECTIVE_NUM; ++j)
    {
        if (g<fabs(individual[index_x_best].f[j] - z[j])/lambda[index_weight_best][j])
        {
            g = fabs(individual[index_x_best].f[j] - z[j])/w[j];
        }
    }
    return g;
}

void neighbor_to_adjust(int i,int *neighbor,struct INDIVIDUAL *individual)
{
    int size = ceil((double)(weight_vector_num-number_of_candidates)/number_of_candidates);//9
    struct SORT_LIST sort_list[weight_vector_num];
    for (int j = 0; j < weight_vector_num; ++j)//算出i向量与其它所有向量的距离
    {
        sort_list[j].euclidian_distance=euclidian_distance_between_two_vectors(lambda[i],lambda[j]);//当前第i个权重向量与其余所有weight_vector_num个权重向量的距离
        sort_list[j].index=j;
    }
    struct SORT_LIST temp;
    for (int i = 0; i < weight_vector_num; ++i)
    {
        for (int j = weight_vector_num-1; j >  i; --j)
        {
            if (sort_list[j].euclidian_distance<sort_list[j-1].euclidian_distance)
            {

                temp=sort_list[j-1];
                sort_list[j-1]=sort_list[j];
                sort_list[j]=temp;
            }
        }
    }
    int counter = 0;
    for (int j = 0; j < weight_vector_num;j++)
    {
        if(counter>=size)
        {
            break;
        }
        if(individual[sort_list[j].index].flag==0)
        {
            neighbor[counter] = sort_list[j].index;
            counter++;
            individual[sort_list[j].index].flag = 1;
        }
    }
    return;
}

void update_weight_vector(struct INDIVIDUAL *individual)
{
    /*更新使得所有向量初始状态都可被吸引*/
    for (int i = 0; i < weight_vector_num; ++i)
    {
        individual[i].flag = 0;
    }
    int size = ceil((double)(weight_vector_num-number_of_candidates)/number_of_candidates);//9
    int neighbor[size];
    int u[number_of_candidates];
    struct INDIVIDUAL array[weight_vector_num];
    for (int j = 0; j < weight_vector_num; ++j)
    {
        array[j] = individual[j];
    }
    struct INDIVIDUAL temp;
    for (int i = 0; i < number_of_candidates; ++i)
    {
        for (int j = weight_vector_num-1; j > i; --j)
        {
            if (array[j].score<array[j-1].score)
            {
                temp = array[j-1];
                array[j-1] = array[j];
                array[j] = temp;
            }
        }
    }
    printf("Index of promising solutions:\t");
    for (int k = 0; k < number_of_candidates; ++k)
    {
        u[k] = array[k].index;
        individual[u[k]].flag = 1;//promising weight不能被吸引
        printf("%d ",u[k]);
    }
    printf("\n");
    for (int i = 0; i < number_of_candidates; ++i)//更新这mu个向量的邻居
    {
        double Tchebycheff_best = Tchebycheff_Function(individual);
        printf("individual[%d].score=:%f %f\n",u[i],individual[u[i]].score,Tchebycheff_best);
        if (individual[u[i]].score < Tchebycheff_best)
        {
            neighbor_to_adjust(u[i],neighbor,individual);
            printf("更新下标为%d的向量的邻居\n",u[i]);
            for (int j = 0; j < size; ++j)
            {
                printf("%d ",neighbor[j]);
            }
            printf("\n");
            for (int j = 0; j < size; ++j)
            {
                for (int k = 0; k < OBJECTIVE_NUM; ++k)
                {
                    lambda[neighbor[j]][k] = lambda[neighbor[j]][k] + step_size*(lambda[u[i]][k] - lambda[neighbor[j]][k]);
                }
            }
        }
        else//剩下所有权重向量都向这一个best weight vector移动
        {
            printf("上一次咨询DM选出的最优解对应的权重向量 lambda[%d] :=(\t",index_weight_best);
            for (int j = 0; j < OBJECTIVE_NUM; ++j)
            {
                printf("%f\t",lambda[index_weight_best][j]);

            }
            printf(")\n");
            printf("将其他99个向量都向这个向量靠齐\n");
            for (int j = 0; j < weight_vector_num; ++j)
            {
                for (int k = 0; k < OBJECTIVE_NUM; ++k)
                {
                    lambda[j][k] = lambda[j][k] + 0.8*step_size*(lambda[index_weight_best][k] - lambda[j][k]);//todo:所有90个向量向上一次咨询最好的向量移动时，慢点
                }
            }
            return;
        }
    }
    return;
}

void saving_FV(struct INDIVIDUAL *individual)//todo:改为任意目标
{
    struct INDIVIDUAL *p;
    FILE *fp=NULL;
    fp=fopen(FOPEN_FILENAME,FOPEN_MODE);
    if (OBJECTIVE_NUM==2)
    {
        fprintf(fp,"%s\n","------------------------------------");
        fprintf(fp,"%s\n","Parameter_settings:");
        fprintf(fp,"%s\t%s\n","Test_Problem:         ",STRING2(TEST_PROBLEM));
        fprintf(fp,"%s\t%d\n","Variable_number:      ",VARIABLE_NUM);
        fprintf(fp,"%s\t%d\n","Objective_number:     ",OBJECTIVE_NUM);
        fprintf(fp,"%s\t%d\n","Population_size:      ",weight_vector_num);
        fprintf(fp,"%s\t%d\n","Iterative_time:       ",ITERATIVE_TIME);
        fprintf(fp,"%s\t%f\n","eta_m:                ",eta_m);
        fprintf(fp,"%s\t%f\n","Mutation_probability: ",MUTATION_PROBABILITY);
        fprintf(fp,"%s\t%d\n","T:                    ",T);
        fprintf(fp,"%s\t%f\n","delta:                ",delta);
        fprintf(fp,"%s\t%d\n","n_r:                  ",n_r);
        fprintf(fp,"%s\t%f\n","CR:                   ",CR);
        fprintf(fp,"%s\t%f\n","F:                    ",F);
        fprintf(fp,"%s\n","------------------------------------");
        fprintf(fp,"%s\t%s\n","f1","f2");
        for (int i = 0; i < weight_vector_num; ++i)
        {
            p=individual+i;
            fprintf(fp,"%f\t%f\n",p->f[0],p->f[1]);
        }
    }
    else if (OBJECTIVE_NUM==3)
    {
//        fprintf(fp,"%s\n","-------------------------------");
//        fprintf(fp,"%s\n","Parameter_settings:");
//        fprintf(fp,"%s\t%s\n","Test_Problem:         ",STRING2(TEST_PROBLEM));
//        fprintf(fp,"%s\t%d\n","K:                    ",K);
//        fprintf(fp,"%s\t%d\n","Variable_number:      ",VARIABLE_NUM);
//        fprintf(fp,"%s\t%d\n","Objective_number:     ",OBJECTIVE_NUM);
//        fprintf(fp,"%s\t%d\n","Population_size:      ",weight_vector_num);
//        fprintf(fp,"%s\t%d\n","Iterative_time:       ",ITERATIVE_TIME);
//        fprintf(fp,"%s\t%f\n","eta_m:                ",eta_m);
//        fprintf(fp,"%s\t%f\n","Mutation_probability: ",MUTATION_PROBABILITY);
//        fprintf(fp,"%s\t%f\n","Alpha_for_DTLZ4:      ",alpha);
//        fprintf(fp,"%s\t%d\n","T:                    ",T);
//        fprintf(fp,"%s\t%f\n","delta:                ",delta);
//        fprintf(fp,"%s\t%d\n","n_r:                  ",n_r);
//        fprintf(fp,"%s\t%f\n","CR:                   ",CR);
//        fprintf(fp,"%s\t%f\n","F:                    ",F);
//        fprintf(fp,"%s\n","-------------------------------");
//        fprintf(fp,"%s\t%s\t%s\n","f1","f2","f3");
        for (int i = 0; i < weight_vector_num; ++i)
        {
            p=individual+i;
            fprintf(fp,"%f\t%f\t%f\n",p->f[0],p->f[1],p->f[2]);
        }
    }
    fclose(fp);
    return;
}
/*** RBF Net ***/
double DM(double *x)
{
    double Tchebycheff = EPS;
    double a[3];
    for (int i = 0; i < OBJECTIVE_NUM; ++i)
    {
        a[i] = x[i];
        if(Tchebycheff<fabs(a[i]-z[i])/w[i])
        {
            Tchebycheff = fabs(a[i]-z[i])/w[i];
        }
    }
    return Tchebycheff;
}

double uniform(double floor, double ceil)
{
    return floor + 1.0*rand() / RAND_MAX * (ceil - floor);
}

double RandomNorm(double mu, double sigma, double floor, double ceil)
{
    double x, prob, y;
    do {
        x = uniform(floor, ceil);
        prob = 1 / sqrt(2 * M_PI*sigma)*exp(-1 * (x - mu)*(x - mu) / (2 * sigma*sigma));
        y = 1.0*rand() / RAND_MAX;
    } while (y>prob);
    return x;
}


void Init_weight(double floor, double ceil)
{
    int i;
    for(i=0;i<SAMPLE_SIZE;i++)
    {
        weight[i] = uniform(floor,ceil);
    }
}

void Init_center()
{
    for(int i=0;i<SAMPLE_SIZE;i++)
    {
        //正则化RBFNN，从训练集样本中采样，作为数据中心
        for (int j = 0; j < OBJECTIVE_NUM; ++j)
        {
            center[i][j] = X[i][j];
        }
    }
}

void Init_delta()//todo:根据数据中心的结果也动态更新方差项
{
//    double dmax;
//    dmax=EPS;
//    for (int i = 0; i < SAMPLE_SIZE-1; ++i)
//    {
//        for (int j = i+1; j < SAMPLE_SIZE; ++j)
//        {
//            if(sqrt(pow(center[i][0]-center[j][0],2)+pow(center[i][1]-center[j][1],2)+pow(center[i][2]-center[j][2],2))>dmax)
//            {
//                dmax = sqrt(pow(center[i][0]-center[j][0],2)+pow(center[i][1]-center[j][1],2)+pow(center[i][2]-center[j][2],2));
//            }
//        }
//    }
    //1.由中心数据的距离得到宽度固定值   2.设置宽度参数为定值，eg:0.8
    for(int i=0;i<SAMPLE_SIZE;i++)
    {
//        sigma[i][0] = dmax/sqrt(2*SAMPLE_SIZE);
//        sigma[i][1] = dmax/sqrt(2*SAMPLE_SIZE);
//        sigma[i][2] = dmax/sqrt(2*SAMPLE_SIZE);
        for (int j = 0; j < OBJECTIVE_NUM; ++j)
        {
            sigma[i][j] = 12;
        }
    }
}

double getOutput(double * x)
{
    double y = 0.0;
    double temp;
    for (int i = 0; i<SAMPLE_SIZE; ++i)
    {
        temp = 0.0;
//        y += weight[i] * exp(-1.0*((x[0] - center[i][0])*(x[0] - center[i][0]) / (2 * sigma[i][0] * sigma[i][0]) + (x[1] - center[i][1])*(x[1] - center[i][1]) / (2 * sigma[i][1] * sigma[i][1]) + (x[2] - center[i][2])*(x[2] - center[i][2]) / (2 * sigma[i][2] * sigma[i][2])));
        for (int j = 0; j < OBJECTIVE_NUM; ++j)
        {
            temp+=(x[j] - center[i][j])*(x[j] - center[i][j]) / (2 * sigma[i][j] * sigma[i][j]);
        }
        temp = exp(-1.0*sqrt(temp));
        y += weight[i] * temp;
    }
    return y;
}

double calSingleError(int i)
{
    double output = getOutput(X[i]);
//    printf("Y[index] - output=%f\n",Y[index] - output);
    return Y[i] - output;
}

//网络均方误差
//这里也会更新 Y[index] - output-->error[i]
double calTotalError()
{
    double rect = 0.0;
    int i;
    for (i = 0; i<SAMPLE_SIZE; ++i)
    {
        error[i] = calSingleError(i);
        rect += error[i] * error[i];
    }
    return rect / 2;
}

//使用梯度下降法最小化均方误差-->BP算法
void updateParam()
{
//    int j,i;
//    for (j = 0; j<SAMPLE_SIZE; ++j)
//    {
//        double delta_weight = 0.0;
//        double sum7 = 0.0;
//        for (i = 0; i<SAMPLE_SIZE; ++i)//使用所有样本-->批量梯度下降法
//        {
////            sum1 += error[i] * exp(-1.0*((X[i][0] - center[j][0])*(X[i][0] - center[j][0]) / (2 * sigma[j][0] * sigma[j][0]) + (X[i][1] - center[j][1])*(X[i][1] - center[j][1]) / (2 * sigma[j][1] * sigma[j][1]) + (X[i][2] - center[j][2])*(X[i][2] - center[j][2]) / (2 * sigma[j][2] * sigma[j][2])))*(X[i][0] - center[j][0]);
////            sum2 += error[i] * exp(-1.0*((X[i][0] - center[j][0])*(X[i][0] - center[j][0]) / (2 * sigma[j][0] * sigma[j][0]) + (X[i][1] - center[j][1])*(X[i][1] - center[j][1]) / (2 * sigma[j][1] * sigma[j][1]) + (X[i][2] - center[j][2])*(X[i][2] - center[j][2]) / (2 * sigma[j][2] * sigma[j][2])))*(X[i][1] - center[j][1]);
////            sum3 += error[i] * exp(-1.0*((X[i][0] - center[j][0])*(X[i][0] - center[j][0]) / (2 * sigma[j][0] * sigma[j][0]) + (X[i][1] - center[j][1])*(X[i][1] - center[j][1]) / (2 * sigma[j][1] * sigma[j][1]) + (X[i][2] - center[j][2])*(X[i][2] - center[j][2]) / (2 * sigma[j][2] * sigma[j][2])))*(X[i][2] - center[j][2]);
////
////            sum4 += error[i] * exp(-1.0*((X[i][0] - center[j][0])*(X[i][0] - center[j][0]) / (2 * sigma[j][0] * sigma[j][0]) + (X[i][1] - center[j][1])*(X[i][1] - center[j][1]) / (2 * sigma[j][1] * sigma[j][1]) + (X[i][2] - center[j][2])*(X[i][2] - center[j][2]) / (2 * sigma[j][2] * sigma[j][2])))*(X[i][0] - center[j][0])*(X[i][0] - center[j][0]);
////            sum5 += error[i] * exp(-1.0*((X[i][0] - center[j][0])*(X[i][0] - center[j][0]) / (2 * sigma[j][0] * sigma[j][0]) + (X[i][1] - center[j][1])*(X[i][1] - center[j][1]) / (2 * sigma[j][1] * sigma[j][1]) + (X[i][2] - center[j][2])*(X[i][2] - center[j][2]) / (2 * sigma[j][2] * sigma[j][2])))*(X[i][1] - center[j][1])*(X[i][1] - center[j][1]);
////            sum6 += error[i] * exp(-1.0*((X[i][0] - center[j][0])*(X[i][0] - center[j][0]) / (2 * sigma[j][0] * sigma[j][0]) + (X[i][1] - center[j][1])*(X[i][1] - center[j][1]) / (2 * sigma[j][1] * sigma[j][1]) + (X[i][2] - center[j][2])*(X[i][2] - center[j][2]) / (2 * sigma[j][2] * sigma[j][2])))*(X[i][2] - center[j][2])*(X[i][2] - center[j][2]);
////            printf("error[%d]=%f\n",i,error[i]);
////            sum7 += error[i] * exp(-1.0*((X[i][0] - center[j][0])*(X[i][0] - center[j][0]) / (2 * sigma[j][0] * sigma[j][0]) + (X[i][1] - center[j][1])*(X[i][1] - center[j][1]) / (2 * sigma[j][1] * sigma[j][1]) + (X[i][2] - center[j][2])*(X[i][2] - center[j][2]) / (2 * sigma[j][2] * sigma[j][2])));
//            sum7 += error[i] * exp(-1.0*sqrt((X[i][0] - center[j][0])*(X[i][0] - center[j][0]) / (2 * sigma[j][0] * sigma[j][0]) + (X[i][1] - center[j][1])*(X[i][1] - center[j][1]) / (2 * sigma[j][1] * sigma[j][1]) + (X[i][2] - center[j][2])*(X[i][2] - center[j][2]) / (2 * sigma[j][2] * sigma[j][2])));
//            //sum7 += exp(-1.0*((X[i][0] - center[j][0])*(X[i][0] - center[j][0]) / (2 * sigma[j][0] * sigma[j][0]) + (X[i][1] - center[j][1])*(X[i][1] - center[j][1]) / (2 * sigma[j][1] * sigma[j][1]) + (X[i][2] - center[j][2])*(X[i][2] - center[j][2]) / (2 * sigma[j][2] * sigma[j][2])));
//        }
////        printf("梯度:%f\t",sum7);
////        delta_center_0 = eta * weight[j] / (sigma[j][0] * sigma[j][0])*sum1;
////        delta_center_1 = eta * weight[j] / (sigma[j][1] * sigma[j][1])*sum2;
////        delta_center_2 = eta * weight[j] / (sigma[j][2] * sigma[j][2])*sum3;
////
////        delta_delta_0 = eta * weight[j] / pow(sigma[j][0], 3)*sum4;
////        delta_delta_1 = eta * weight[j] / pow(sigma[j][1], 3)*sum5;
////        delta_delta_2 = eta * weight[j] / pow(sigma[j][2], 3)*sum6;
////        printf("sum7=%f\n",sum7);
//        delta_weight = eta * sum7;//sum>0
////        printf("权重参数改变量：%f\n",delta_weight);
////        center[j][0] += delta_center_0;
////        center[j][1] += delta_center_1;
////        center[j][2] += delta_center_2;
////
////        sigma[j][0] += delta_delta_0;
////        sigma[j][1] += delta_delta_1;
////        sigma[j][2] += delta_delta_2;
//
//        weight[j] += delta_weight;//化简过了
////        printf("weight[%d]=%f\n",j,weight[j]);
//    }
    Eigen::MatrixXd e(SAMPLE_SIZE,SAMPLE_SIZE);
    double temp;
    for (int k = 0; k < SAMPLE_SIZE; ++k)//控制样本号
    {
        for (int l = 0; l < SAMPLE_SIZE; ++l)//控制基函数中心号
        {
            temp = 0.0;
            for (int i = 0; i < OBJECTIVE_NUM; ++i)
            {
                temp+=(X[k][i] - center[l][i])*(X[k][i] - center[l][i]) / (2 * sigma[l][i] * sigma[l][i]);
            }
            e(l,k) = exp(-1.0*sqrt(temp));
        }
    }

    Eigen::MatrixXd out(1,SAMPLE_SIZE);
    for (int i = 0; i < SAMPLE_SIZE; ++i)
    {
        out(0,i) = Y[i];
    }

    Eigen::MatrixXd w_;
    w_ = out*pinv_eigen_based(e);
    for (int i = 0; i < SAMPLE_SIZE; ++i)
    {
        weight[i] = w_(0,i);
    }
}
