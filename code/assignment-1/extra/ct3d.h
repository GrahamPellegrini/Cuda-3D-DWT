/*
    \file ct3d.h
    Header file for a 3D image class to be used in the CT DWT implementation.
    This class represents a simple 3D image with basic functionalities.

    Author: Graham Pellegrini
    Date: 10/1/2019

    This file is part of CCE3015 Assignment 1.

    Build included cited:
    \file jbulit.h 
    Copyright (c) 2011-2014 Johann A. Briffa
    Licensed under the GNU General Public License v3.0 or later.
*/
#include <iostream>
#include <vector>
#include <string>
#include <iomanip> 
#include <sstream> 
#include "../shared/jbutil.h"

template <class T>
class image3D {
private:
    std::vector<jbutil::image<T>> m_data; // Vector of 2D images in jbulit namespace
    int m_depth; // Depth (number of 2D slices)
public:
    // Constructor
    explicit image3D(int depth = 0, int rows = 0, int cols = 0, int maxval = 255)
        : m_depth(depth), m_data(depth, jbutil::image<T>(rows, cols, 1, maxval)) // Initialize vector with images
    {
    }

    // Access methods
    int get_depth() const { return m_depth; }
    int get_rows() const {
        return m_depth > 0 ? m_data[0].get_rows() : 0; // Check depth before accessing
    }
    int get_cols() const {
        return m_depth > 0 ? m_data[0].get_cols() : 0; // Check depth before accessing
    }

    jbutil::image<float>& get_slice(int index) {
    return m_data[index];
    }

    // Access pixel (c = channel, z = depth, i = row, j = col)
    T& operator()(int z, int i, int j) {
        assert(z >= 0 && z < m_depth);
        return m_data[z](0, i, j); // Assuming single channel for each 2D slice
    }

    const T& operator()(int z, int i, int j) const {
        assert(z >= 0 && z < m_depth);
        return m_data[z](0, i, j);
    }
    
    // Load images from a directory from multiple DICOM files
    bool load_directory(const std::string& directory, int depth)
    {  
        m_depth = depth;
        m_data.resize(depth);

        for (int z = 0; z < depth; ++z) {
            jbutil::image<T> img;

            // Generate the zero-padded filename
            std::string padding = std::string(4 - std::to_string(z).length(), '0');
            std::string filename = directory + "i" + padding + std::to_string(z) + ",0000b.dcm"; // Store filename as string

            // Open the file using std::ifstream
            std::ifstream file(filename, std::ios::binary); // Open the file as a binary stream

            // Check if the file was successfully opened
            if (!file) {
                std::cerr << "Error loading DICOM file: " << filename << std::endl;
                return false; // Return false if the file couldn't be opened
            }

            // Load the image (assuming img.load() works with std::ifstream or file stream)
            img.load(file);

            // Store the loaded image
            m_data[z] = img;
        }
    }

    // Save images to a directory to multiple DICOM files
    bool save_directory(const std::string& directory) const {
        for (int z = 0; z < m_depth; ++z) {
            // Create a zero-padded filename
            std::ostringstream filename;
            filename << directory << "i" << std::setw(4) << std::setfill('0') << z << ",0000b.dcm";

            // Attempt to save the image
            if (!m_data[z].save(filename.str())) {
                std::cerr << "Error saving DICOM file: " << filename.str() << std::endl;
                return false; // Return false if saving fails
            }
        }

        return true; // Return true if saving was successful
    }
};

