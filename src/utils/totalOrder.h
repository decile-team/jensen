/*
 * totalOrder.h
 *
 *  A total order is a permutation of [n]
 *  Created on: Nov 13, 2011
 *      Author: Hui Lin (hlin@ee.washington.edu)
 */
#include <algorithm>
#include <vector>
#ifndef TOTALORDER_H_
#define TOTALORDER_H_

template<class ValueType>
class totalOrder {
public:
  typedef std::vector<int>::const_iterator order_iterator;

  struct pair_less {
    bool operator ()(std::pair<ValueType,int> const& a,
                 std::pair<ValueType,int> const& b) const {
            if (a.first < b.first) return true;
             return false;
      }
  };

  totalOrder(){
    _order.clear();
  }

  totalOrder(int n){
    _order.clear();
    for(int i = 0; i < n; i++)
      _order.push_back(i);
  }

  /////////////////////////////////////////////////////////////////////////
  // generate a total order such that for all i \prec j, w[i] <= w[j]    //
  /////////////////////////////////////////////////////////////////////////
  totalOrder(std::vector<ValueType> w, bool Inc){
	  if(Inc)
    	  resetOrderInc(w);
	  else
		  resetOrderDec(w);
  }

  void resetOrderInc(std::vector<ValueType> w){
    std::vector< std::pair<ValueType,int> > wp;
    wp.clear();
    for(int i=0;i<w.size();++i){
       std::pair<ValueType,int> myw;
       myw.first = w[i];
       myw.second = i;
       wp.push_back(myw);
    }

    std::sort(wp.begin(),wp.end(),pair_less());

    _order.clear();
    for(int i = 0; i < w.size(); i++)
      _order.push_back(wp[i].second);
  }

  /////////////////////////////////////////////////////////////////////////
  // generate a total order such that for all i \prec j, w[i] >= w[j]    //
  /////////////////////////////////////////////////////////////////////////
  
  void resetOrderDec(std::vector<ValueType> w){
    std::vector< std::pair<ValueType,int> > wp;
    wp.clear();
    for(int i=0;i<w.size();++i){
       std::pair<ValueType,int> myw;
       myw.first = -1*w[i];
       myw.second = i;
       wp.push_back(myw);
    }

    std::sort(wp.begin(),wp.end(),pair_less());

    _order.clear();
    for(int i = 0; i < w.size(); i++)
      _order.push_back(wp[i].second);
  }

  void randomizeOrder(int n){
    _order.clear();
    for(int i = 0; i < n; i++)
      _order.push_back(i);
    std::random_shuffle(_order.begin(),_order.end());
  }

  void push_back(int i){ _order.push_back(i); }

  //////////////////////////////////////////////////////////////////////////
  // implementation of access functions.                                  //
  //////////////////////////////////////////////////////////////////////////
  int order(int i) { return _order[i]; }
  int size() { return _order.size(); }
  void clear() { _order.clear(); }

    inline int operator()(unsigned int i) {
      return _order[i];
    }
    inline int operator[](unsigned int i) {
      return _order[i];
    }

protected:
  std::vector<int> _order;

};

#endif /* TOTALORDER_H_ */

