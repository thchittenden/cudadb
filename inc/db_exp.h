#ifndef _DB_EXP_H_
#define _DB_EXP_H_

struct record {
	int id;
	long time;
};

#ifndef TABLE
#define TABLE(x) 
#endif 
#ifndef CRITERIA
#define CRITERIA(x, y)
#endif

TABLE(record);
CRITERIA(record, 2);

#endif
