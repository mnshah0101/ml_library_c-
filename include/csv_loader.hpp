#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>
#include <stdexcept>
#include <algorithm>

class CSVLoader { 
    public:
        CSVLoader(const std::string& filename, char delimiter = ',')
            : m_filename{filename}, m_delimiter{delimiter}
        {
        }

        void load(){
            std::ifstream file(m_filename);
            if (!file.is_open())
            {
                throw std::runtime_error("Could not open file: " + m_filename);
            }
            m_data.clear();
            std::string line;

            while (std::getline(file, line))
            {
                m_data.push_back(parseLine(line));
            }

            // Create column name to index mapping
            if (!m_data.empty()) {
                const auto& header = m_data[0];
                for (size_t i = 0; i < header.size(); ++i) {
                    m_column_indices[trim(header[i])] = i;
                }
            }
        }

        const std::vector<std::vector<std::string>> &getData() const
        {
            return m_data;
        }

        // Get all column names from the header
        std::vector<std::string> getColumnNames() const {
            if (m_data.empty()) {
                throw std::runtime_error("No data loaded. Call load() first.");
            }
            std::vector<std::string> trimmed_names;
            for (const auto& name : m_data[0]) {
                trimmed_names.push_back(trim(name));
            }
            return trimmed_names;
        }

        // Get the index of a column by name
        int getColumnIndex(const std::string& column_name) const {
            auto it = m_column_indices.find(trim(column_name));
            if (it == m_column_indices.end()) {
                throw std::runtime_error("Column not found: " + column_name);
            }
            return it->second;
        }

        // Check if a column exists
        bool hasColumn(const std::string& column_name) const {
            return m_column_indices.find(trim(column_name)) != m_column_indices.end();
        }

    private:
        std::string m_filename;
        char m_delimiter{};
        std::vector<std::vector<std::string>> m_data;
        std::map<std::string, int> m_column_indices;  // Maps column names to their indices

        // Helper function to trim whitespace from both ends of a string
        static std::string trim(const std::string& str) {
            const std::string whitespace = " \t\r\n";
            size_t start = str.find_first_not_of(whitespace);
            if (start == std::string::npos) {
                return "";  // String is all whitespace
            }
            size_t end = str.find_last_not_of(whitespace);
            return str.substr(start, end - start + 1);
        }

        std::vector<std::string> parseLine(const std::string &line)
        {
            std::vector<std::string> result;
            std::string field;
            std::istringstream stream(line);
            bool inQuotes = false;
            char c;

            while (stream.get(c))
            {
                if (c == '"')
                {
                    inQuotes = !inQuotes;
                }
                else if (c == m_delimiter && !inQuotes)
                {
                    result.push_back(field);
                    field.clear();
                }
                else
                {
                    field.push_back(c);
                }
            };
            result.push_back(field);
            return result;
        };
}; 