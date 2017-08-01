#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  
  num_particles = 20;
  
  particles.clear();
  weights.clear();
  
  // Creating normal distributions for the particle position and heading
  // with a mean = 0 and standard deviation = GPS measurement uncertainty
  normal_distribution<double> x_distribution(x, std[0]);
  normal_distribution<double> y_distribution(y, std[1]);
  normal_distribution<double> theta_distribution(theta, std[2]);
  
  // Creating a random seed generator for the random number generator
  random_device seed;
  
  // Creating a random number generator
  mt19937 generator(seed());
  
  // Initializing particles
  for (int i = 0; i < num_particles; ++i) {
    
    Particle single_particle;
    
    single_particle.id = i;
    
    // Sampling from the normal distributions using the random number generator
    single_particle.x = x_distribution(generator);
    single_particle.y = y_distribution(generator);
    single_particle.theta = theta_distribution(generator);
    
    single_particle.weight = 1.0;
    
    particles.push_back(single_particle);
    
    // Creating a separate list of weights later used for resampling
    weights.push_back(1.0);
    
  }
  
  is_initialized = true;
  
}


void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
  
  // Creating a random seed generator for the random number generator
  random_device seed;
  
  // Creating a random number generator
  mt19937 generator(seed());
  
  for (int i = 0; i < num_particles; ++i) {
    
    double x_updated, y_updated, theta_updated;
    
    x_updated = 0;
    y_updated = 0;
    theta_updated = 0;
    
    // Updating the particle position and heading
    
    if (fabs(yaw_rate) < 0.000001) {
      
      x_updated = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      y_updated = particles[i].y + velocity * delta_t * sin(particles[i].theta);
      theta_updated = particles[i].theta;
      
    }
    
    else if (fabs(yaw_rate) > 0) {
      
      x_updated = (particles[i].x + (velocity / yaw_rate)
                   * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)));
      y_updated = (particles[i].y + (velocity / yaw_rate)
                   * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)));
      theta_updated = particles[i].theta + yaw_rate * delta_t;
      
    }
    
    // Creating normal distributions for the particle position and heading
    // with a mean = 0 and standard deviation = GPS measurement uncertainty
    normal_distribution<double> x_distribution(x_updated, std_pos[0]);
    normal_distribution<double> y_distribution(y_updated, std_pos[1]);
    normal_distribution<double> theta_distribution(theta_updated, std_pos[2]);
    
    // Sampling from the normal distributions using the random number generator
    particles[i].x = x_distribution(generator);
    particles[i].y = y_distribution(generator);
    particles[i].theta = theta_distribution(generator);
    
  }
  
}


double ParticleFilter::calculateDistance(double x1, double y1, double x2, double y2) {
  
  return sqrt(pow((x2 - x1), 2.0) + pow((y2 - y1), 2.0));
  
}


void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   std::vector<LandmarkObs> observations, Map map_landmarks) {
  
  double distance, bivariate_gaussian, x_std_dev, y_std_dev;

  bivariate_gaussian = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
  
  x_std_dev = 2 * pow(std_landmark[0], 2.0);
  
  y_std_dev = 2 * pow(std_landmark[1], 2.0);
  
  for (int i = 0; i < num_particles; ++i) {
    
    particles[i].associations.clear();
    particles[i].sense_x.clear();
    particles[i].sense_y.clear();
    
    // Creating a list of map landmarks in range of particle
    vector<Map::single_landmark_s> landmarks_in_range;
    landmarks_in_range.clear();

    for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {
      
      distance = calculateDistance(particles[i].x, particles[i].y,
                                   map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
      
      // Comparing calculated distance between particle and landmark with maximum sensor range
      if (distance <= sensor_range) {
        landmarks_in_range.push_back(map_landmarks.landmark_list[j]);
      }
      
    }
    
    // Transforming observed landmarks from vehicle to map coordinates
    vector<LandmarkObs> transformed_observations;
    transformed_observations.clear();
    
    for (int k = 0; k < observations.size(); ++k) {
      
      LandmarkObs transformed_observation;

      transformed_observation.x = (particles[i].x + observations[k].x * cos(particles[i].theta)
                                   - observations[k].y * sin(particles[i].theta));
      transformed_observation.y = (particles[i].y + observations[k].y * cos(particles[i].theta)
                                   + observations[k].x * sin(particles[i].theta));
      
      // Associating nearest map landmark to observed landmark
      transformed_observation.id = -1;
      
      double best_distance = sensor_range;
      int best_index;
      
      for (int m = 0; m < landmarks_in_range.size(); ++m) {
        
        distance = calculateDistance(transformed_observation.x, transformed_observation.y,
                                     landmarks_in_range[m].x_f, landmarks_in_range[m].y_f);
        
        if (distance < best_distance) {
          
          // Saving the index of the nearest landmark
          best_index = m;
          best_distance = distance;
          
        }
        
      }
      
      transformed_observation.id = best_index;
      
      transformed_observations.push_back(transformed_observation);
      
      particles[i].associations.push_back(landmarks_in_range[best_index].id_i);
      particles[i].sense_x.push_back(transformed_observation.x);
      particles[i].sense_y.push_back(transformed_observation.y);
      
    }
    
    // Updating particle weight
    double updated_weight = 1.0;
    
    for (int n = 0; n < transformed_observations.size(); ++n) {
      
      // Checking if a map landmark was assigned to an observed landmark
      if (transformed_observations[n].id != -1) {
        
        // Extracting values from vectors for clarity
        double x_measurement, y_measurement, x_mean, y_mean;
        
        // Measurements
        x_measurement = transformed_observations[n].x;
        y_measurement = transformed_observations[n].y;
        
        // Predicted measurements
        x_mean = landmarks_in_range[transformed_observations[n].id].x_f;
        y_mean = landmarks_in_range[transformed_observations[n].id].y_f;
        
        updated_weight *= bivariate_gaussian * exp(-((pow((x_measurement - x_mean), 2.0) / x_std_dev) + (pow((y_measurement - y_mean), 2.0) / y_std_dev)));
        
      }
      
    }
    
    // Checking if the weight was updated
    if (updated_weight != 1.0) {
      
      particles[i].weight = updated_weight;
      weights[i] = updated_weight;
      
    }
    
  }
  
}


void ParticleFilter::resample() {
  
  // Creating a vector of resampled particles
  vector<Particle> resampled_particles;
  resampled_particles.clear();
  
  // Creating a random seed generator for the random number generator
  random_device seed;
  
  // Creating a random number generator
  mt19937 generator(seed());
  
  // Creating a discrete distribution based on the weights
  discrete_distribution<> weight_distribution(weights.begin(), weights.end());
  
  // Resampling particles with replacement based on the discrete distribution
  for (int i = 0; i < num_particles; ++i) {

    resampled_particles.push_back(particles[weight_distribution(generator)]);
  }
  
  // Assigning the resampled particles to the current list of particles
  particles = resampled_particles;

}


string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}


string ParticleFilter::getSense(Particle best, std::string coordinate) {
  
  vector<double> v;
  
  if (coordinate == "sense_x") {
    v = best.sense_x;
  }
  
  else if (coordinate == "sense_y") {
    v = best.sense_y;
  }
  
  stringstream ss;
  copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
