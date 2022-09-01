
#include "progress_bar.hpp"

ProgressBar::ProgressBar(int total, int char_len) :
    total(total), value(0), char_len(char_len)
{}

void ProgressBar::set(int new_value)
{
    value = new_value;
    display();
}

void ProgressBar::inc()
{
    ++value;
    display();
}

void ProgressBar::display()
{
    std::string bar = "[";
    int filled_chars = (static_cast<double>(value) / total) * char_len;
    for (int i = 0; i <= filled_chars; ++i)
    {
        bar += "=";
    }
    if (filled_chars != char_len) bar += ">";
    for (int i = filled_chars + 1; i < char_len; ++i)
    {
        bar += " ";
    }
    bar += "]";
    BUNJI_LOG_REPLACE(fg(fmt::color::white),"{}", bar);
}

void ProgressBar::end()
{
    
}