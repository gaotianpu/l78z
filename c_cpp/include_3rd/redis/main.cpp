#include <stdio.h>
#include <sqlite3.h>

int main(int argc, char *argv[])
{
    sqlite3 *db;
    char *zErrMsg = 0;
    int rc;

    rc = sqlite3_open("test.db", &db);

    if (rc)
    {
        fprintf(stderr, "Can't open database: %s\n", sqlite3_errmsg(db));
        // exit(0);
        return 0;
    }
    else
    {
        fprintf(stderr, "Opened database successfully\n");
    }
    sqlite3_close(db);
    return 1 ;
}

// errors
// g++ main.cpp -l sqlite3
// g++ -std=c++17 -o main main.cpp  -lsqlite3
// g++ -std=c++17 -o main main.cpp -I /mnt/d/anconda3/include -l sqlite3 
// ok 
// g++ -std=c++17 -o main main.cpp -I /mnt/d/anconda3/include -L /mnt/d/anconda3/lib -l sqlite3 