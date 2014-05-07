#include "db.h"
#include "db_exp.h"
#include "db_insert.h"
#include "db_select.h"
#include "db_criteria.h"

#include "CycleTimer.h"

#include <iostream>
#include <thread>
using namespace std;

#define NUM 1000

int main() {
	db_init(10*1024*1024);
	auto t = db_table_create(&record::id, &record::time); 
	cout << "got table: " << t << endl;

	record* recs = new record[NUM];
	for(int i = 0; i < NUM; i++) {
		recs[i].id = i % 20;
		recs[i].time = i;
	}

	double pre_insert = CycleTimer::currentSeconds();
	db_insert_bulk(t, recs, NUM);
	double post_insert = CycleTimer::currentSeconds();

	// allow insert to finish
	this_thread::sleep_for(chrono::seconds(2));

	double pre_prepare = CycleTimer::currentSeconds();
	auto h = db_select_prepare(t, EQ(&record::id, 2), EQ(&record::id, 2));
	double post_prepare = CycleTimer::currentSeconds();

	double avg_next = 0.0f;
	double max_next = 0.0f;
	int max_iter = 0;
	int cur_iter = 0;
	record* rec;
	do {
		
		double pre_next = CycleTimer::currentSeconds();
		rec = db_select_next(h);
		double next_time = CycleTimer::currentSeconds() - pre_next;
		if(next_time > max_next) {
			max_next = next_time;
			max_iter = cur_iter;
		}
		if(rec != NULL) {
			cout << "got rec  {id = " << rec->id << ", time = " << rec->time << "}" << endl;
		} else {
			cout << "no rec  found" << endl;
		}
		avg_next += next_time;
		cur_iter++;

	} while(rec != NULL);
	avg_next /= cur_iter;

	cout << "insert time: " << 1000.f * (post_insert - pre_insert) << endl;
	cout << "prepare time: " << 1000.f * (post_prepare - pre_prepare) << endl;
	cout << "avg select time: " << 1000.f * avg_next << endl;
	cout << "max select time: " << 1000.f * max_next << " on iteration " << max_iter << endl;

	db_select_destroy(h);
	db_table_destroy(t);
}
