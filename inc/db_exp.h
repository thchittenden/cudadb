#ifndef _DB_EXP_H_
#define _DB_EXP_H_

struct record {
	int id;
	long time;
};

#ifndef TABLE
#define TABLE(x) 
#endif 

TABLE(record);

#endif
