#include "log.hpp"

class ProgressBar
{
private:
    int total;
    int value;
    int char_len;
public:
    ProgressBar(int total, int char_len);
    void set(int value);
    void inc();
    void display();
    void end();
};