/* clib.cpp */
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <boost/pending/disjoint_sets.hpp>
#include <vector>
#include <queue>
#include <map>
using namespace std;

template <class T>
class AffinityGraphCompare{
	private:
	    const T * mEdgeWeightArray;
	public:
		AffinityGraphCompare(const T * EdgeWeightArray){
			mEdgeWeightArray = EdgeWeightArray;
		}
		bool operator() (const int& ind1, const int& ind2) const {
			return (mEdgeWeightArray[ind1] > mEdgeWeightArray[ind2]);
		}
};

    /*
     * Compute the MALIS loss function and its derivative wrt the affinity graph
     * MAXIMUM spanning tree
     * Author: Srini Turaga (sturaga@mit.edu)
     * All rights reserved
     */
     void malisLoss(const int* dims, const float* conn, const double* nhood, const int* seg, const double margin,
     const bool pos, float* losses, double* lossReturn, double* randIndexReturn){
//        cout << "malis setup..." << endl;
     	/* input arrays */
//     	cout << "dims: "<<dims[0]<<" "<<dims[1]<<" "<<dims[2]<<" "<<dims[3]<<endl;
//     	cout << "marginArg: "<<marginArg<< " posArg: "<<posArg<<endl;
         // 4d connectivity graph [y * x * z * #edges]
       
     	const int	conn_num_dims		= 4;
     	const int*	conn_dims			= dims;
     	const float*	conn_data		= conn;
         // graph neighborhood descriptor [3 * #edges]
     	const int nhoodDims[2]          = {dims[3],dims[3]};
     	const int*	nhood_dims			= &nhoodDims[0];
     	const double*	nhood_data		= nhood;
         // true target segmentation [y * x * z]
     	const int*	seg_data			= seg;
         // sq-sq loss margin [0.3]
//         const double    margin         = marginArg;
         // is this a positive example pass [true] or a negative example pass [false] ?
//         const bool      pos            = posArg;

         /* Output arrays */
         // the derivative of the MALIS-SqSq loss function
         // (times the derivative of the logistic activation function) [y * x * z * #edges]
         float* dloss_data = losses;

        
        /* Cache for speed to access neighbors */
        int nVert = 1;
        for (int i=0; i<conn_num_dims-1; ++i)
            nVert = nVert*conn_dims[i];

        vector<int> prodDims(conn_num_dims-1); prodDims[0] = 1;
        for (int i=1; i<conn_num_dims-1; ++i)
            prodDims[i] = prodDims[i-1]*conn_dims[i-1];

         /* convert n-d offset vectors into linear array offset scalars */
        vector<int> nHood(nhood_dims[0]);
        for (int i=0; i<nhood_dims[0]; ++i) {
            nHood[i] = 0;
            for (int j=0; j<nhood_dims[1]; ++j) {
                nHood[i] += (int)nhood_data[i+j*nhood_dims[0]] * prodDims[j];
            }
        }

        /* Disjoint sets and sparse overlap vectors */
        vector<map<int,int> > overlap(nVert);
        vector<int> rank(nVert);
        vector<int> parent(nVert);
        map<int,int> segSizes;
        unsigned long nLabeledVert=0;
        unsigned long nPairPos=0;
        boost::disjoint_sets<int*, int*> dsets(&rank[0],&parent[0]);
        for (int i=0; i<nVert; ++i){
            dsets.make_set(i);
            if (0!=seg_data[i]) {
                overlap[i].insert(pair<int,int>(seg_data[i],1));
                ++nLabeledVert;
                ++segSizes[seg_data[i]];
                nPairPos +=(unsigned long) (segSizes[seg_data[i]]-1);
            }
        }

        unsigned long nPairTot = (nLabeledVert*(nLabeledVert-1))/2;
        unsigned long nPairNeg = nPairTot - nPairPos;
        unsigned long nPairNorm;
        if (pos) {nPairNorm = nPairPos;} else {nPairNorm = nPairNeg;}
//        cout << endl;
//        cout << "pos: " << pos << endl;
//        cout << "nPairPos: " << nPairPos << " " << endl;
//        cout << "nPairNorm: " << nPairNorm<<endl;
        /* Sort all the edges in increasing order of weight */
        std::vector< int > pqueue( static_cast< int >(3) *
                                       ( conn_dims[0]-1 ) *
                                       ( conn_dims[1]-1 ) *
                                       ( conn_dims[2]-1 ));
        int j = 0;
        for ( int d = 0, i = 0; d < conn_dims[3]; ++d )
            for ( int z = 0; z < conn_dims[2]; ++z )
                for ( int y = 0; y < conn_dims[1]; ++y )
                    for ( int x = 0; x < conn_dims[0]; ++x, ++i )
                    {
                        if ( x > 0 && y > 0 && z > 0 )
                            pqueue[ j++ ] = i;
                    }

        sort( pqueue.begin(), pqueue.end(), AffinityGraphCompare<float>( conn_data ) );


//        cout << "malis MST..." << endl;
//        cout << "pqueue.size: " << pqueue.size() << endl;
        /* Start MST */
        int minEdge;
        int e, v1, v2;
        int set1, set2;
        int nPair = 0;
        double loss=0, dl=0;
        unsigned int nPairIncorrect = 0;
        map<int,int>::iterator it1, it2;
    
        /* Start Kruskal's */
        for (unsigned int i = 0; i < pqueue.size(); ++i ) {
            //cerr << i << endl;
            minEdge = pqueue[i];
            e = minEdge/nVert; v1 = minEdge%nVert; v2 = v1+nHood[e];
    
            //cerr << "v1: " << v1 << endl;
            set1 = dsets.find_set(v1);
            //cerr << "v2: " << v2 << endl;
            set2 = dsets.find_set(v2);
            //cerr << "dsets found\n";

            if (set1!=set2){
                dsets.link(set1, set2);
    
                /* compute the dloss for this MST edge */
                for (it1 = overlap[set1].begin();
                        it1 != overlap[set1].end(); ++it1) {
                    for (it2 = overlap[set2].begin();
                            it2 != overlap[set2].end(); ++it2) {
    
                        nPair = it1->second * it2->second;
    
                        if (pos && (it1->first == it2->first)) {
                            // +ve example pairs
                            // Sq-Sq loss is used here
                            dl = max(0.0,.5+margin-conn_data[minEdge]);
                            loss += .5*dl*dl*nPair;
                            dloss_data[minEdge] += dl*nPair;
                            if (conn_data[minEdge] <= .5) { // an error
                                nPairIncorrect += (unsigned int) (nPair);
                            }
    
                        } else if ((!pos) && (it1->first != it2->first)) {
                            // -ve example pairs
                            // Sq-Sq loss is used here
                            dl = -max(0.0,conn_data[minEdge]-.5+margin);
                            loss += .5*dl*dl*nPair;
                            dloss_data[minEdge] += dl*nPair;
                            if (conn_data[minEdge] > .5) { // an error
                                nPairIncorrect += (unsigned int) (nPair);
                            }
                        }
                    }
                }
                dloss_data[minEdge] /= (float)(nPairNorm);
                /* HARD-CODED ALERT!!
                 * The derivative of the activation function is also multiplied here.
                 * Assumes the logistic nonlinear activation function.
                 */
                dloss_data[minEdge] *= conn_data[minEdge]*(1-conn_data[minEdge]); // DSigmoid
    
                /* move the pixel bags of the non-representative to the representative */
                if (dsets.find_set(set1) == set2) // make set1 the rep to keep and set2 the rep to empty
                    swap(set1,set2);
    
                it2 = overlap[set2].begin();
                while (it2 != overlap[set2].end()) {
                    it1 = overlap[set1].find(it2->first);
                    if (it1 == overlap[set1].end()) {
                        overlap[set1].insert(pair<int,int>(it2->first,it2->second));
                    } else {
                        it1->second += it2->second;
                    }
                    overlap[set2].erase(it2++);
                }
            } // end link
    
        } // end while

//        cout << "malis return..." << endl;
//        cout << "nPairIncorrect:\t\t" << nPairIncorrect << endl;
//        cout << "nPairNorm:\t\t\t" << nPairNorm << endl;
        /* Return items */
        //loss /= (double)nPairNorm; //this should be done outside the malis function
        *lossReturn = loss;
        *randIndexReturn = 1.0 - ((double)nPairIncorrect / (double)nPairNorm);
     }
