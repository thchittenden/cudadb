#include "db.h"
#include "db_exp.h"
#include "db_select.h"

#include <iostream>
using namespace std;

int main() {
	db_init(500*1024*1024);
	auto t = db_table_create(&record::id, &record::time); 
	cout << "got table: " << t << endl;

	select_handle<record, 2>* h = db_select_prepare(t, make_pair(&record::id, 10), make_pair(&record::time, (long)100));
	cout << "got handle: " << h << endl;

	record* rec;
	while((rec = db_select_next(h)) != NULL) {
		cout << "got rec: " << rec << endl;
	}
	db_select_destroy(h);
	db_table_destroy(t);
}
