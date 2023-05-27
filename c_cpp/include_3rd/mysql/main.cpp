#include <mysql++.h>
#include <iostream>
  
using namespace std;
int main()
{
  const char* db = 0, *server = 0, *user = 0, *password = "";
  db = "test";
  server = "localhost";
  user = "root";
  password = "1234567";
     
  mysqlpp::Connection conn(false);
  if (conn.connect(db, server, user, password)) 
  {
    cout << "connect db succeed. " << endl;
    mysqlpp::Query query = conn.query("SELECT * FROM Student");
    if (mysqlpp::StoreQueryResult res = query.store()) 
    {
      cout.setf(ios::left);
      cout << setw(31) << "Sid" <<
      setw(10) << "Sname" <<
      setw(10) << "Sage" <<
      setw(10) << "Sgender" <<                 
      setw(10) << "SDepartment" << endl;
 
      mysqlpp::StoreQueryResult::const_iterator it;
      for (it = res.begin(); it != res.end(); ++it) 
      {
        mysqlpp::Row row = *it;
        cout << setw(30) << row[0] << ' ' <<
        setw(9) << row[1] << ' ' <<
        setw(9) << row[2] << ' ' <<
        setw(9) << row[3] << ' ' <<
        setw(9) << row[4] << ' ' <<
        endl;
      }
    }
  } 
  else 
  {
      cout << "connect db fail. " << endl;
  }
  return 0;
}