#include "db.h"
#include "db_exp.h"
#include "db_insert.h"
#include "db_select.h"

#include <iostream>
using namespace std;

int main() {
	db_init(500*1024*1024);
	auto t = db_table_create(&record::id, &record::time); 
	cout << "got table: " << t << endl;

	record* recs = new record[1000];
	for(int i = 0; i < 1000; i++) {
		recs[i].id = 10;
		recs[i].time = 1000+i;
	}
	db_insert_bulk(t, recs, 1000);

	auto h = db_select_prepare(t, make_pair(&record::id, 10)); 
	record* rec;
	while((rec = db_select_next(h)) != NULL) {
		cout << "got record{id=" << rec->id << ", time=" << rec->time << "}" << endl;
	}
	
	db_select_destroy(h);
	db_table_destroy(t);
}
