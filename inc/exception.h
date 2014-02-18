#pragma once
#include <boost/exception/exception.hpp>


namespace  expt{

enum serverity { warning, error };

}

class p_exception : virtual public boost::exception {
 public:
	 p_exception() = default;
	 p_exception(const std::string& msg, expt::serverity s);
	 ~p_exception() {};

	virtual const char* what() const throw();
	
 private:
	 std::string msg;
};
