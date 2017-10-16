/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

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

#define EPS 0.00001

using namespace std;

static default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 200;
	weights.resize(num_particles, 1.0f);

	// Normal distributions simulating sensor noise
	normal_distribution<double> n_dist_x(0, std[0]);
	normal_distribution<double> n_dist_y(0, std[1]);
	normal_distribution<double> n_dist_theta(0, std[2]);

	for (int i=0;i<num_particles;++i){
		// New particle
		Particle particle;

		// Initialize values
		particle.id = i;
		particle.x = x;
		particle.y = y;
		particle.theta = theta;
		particle.weight = 1.0;

		// Add random Gaussian noise to x,y,theta
		particle.x += n_dist_x(gen);
		particle.y += n_dist_y(gen);
		particle.theta += n_dist_theta(gen);

		// Add particle to vector
		particles.push_back(particle);
	}

	 is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	// Normal distributions simulating sensor noise
	normal_distribution<double> n_dist_x(0, std_pos[0]);
	normal_distribution<double> n_dist_y(0, std_pos[1]);
	normal_distribution<double> n_dist_theta(0, std_pos[2]);

	// For each particle predict x,y,theta
	for (int i=0;i<num_particles;++i){

		if (fabs(yaw_rate) > EPS){
			particles[i].x += (velocity/yaw_rate)*(sin(particles[i].theta+yaw_rate*delta_t)-sin(particles[i].theta));
			particles[i].y += (velocity/yaw_rate)*(cos(particles[i].theta) - cos(particles[i].theta+yaw_rate*delta_t));
			particles[i].theta += yaw_rate*delta_t;
		} else {
			particles[i].x += velocity * delta_t * cos(particles[i].theta);
			particles[i].y += velocity * delta_t * sin(particles[i].theta);
		}

		// Add random Gaussian noise to x,y,theta
		particles[i].x += n_dist_x(gen);
		particles[i].y += n_dist_y(gen);
		particles[i].theta += n_dist_theta(gen);


	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for (int i=0;i<observations.size();++i){

		LandmarkObs obs = observations[i];
		int predLandmarkId = -1;
		double minDistance = INFINITY;

		// Loop in every predicted landmark and compute the min distance from the observation selected
		for (int j=0;j<predicted.size();++j){
			// The current predicted landmark
			LandmarkObs pred = predicted[j];
			// Compute distance between predicted and observed landmark
			double distance = dist(obs.x,obs.y, pred.x,pred.y);
			// Find the nearest neighbor predicted landmark from the observed landmark
			if (distance<minDistance){
				minDistance = distance;
				predLandmarkId = pred.id;
			}
		}
		// Associate landmark id with observed id
		observations[i].id = predLandmarkId;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html


	// Loop each particle
	for (int i=0;i<num_particles;++i){
		Particle p = particles[i];
		// Particle coordinates
		double p_x=p.x;
		double p_y=p.y;
		double p_theta=p.theta;

		// Vector to hold landmarks within the range of the particle
		vector<LandmarkObs> predLandmarkInRange;

		for (int j=0;j<map_landmarks.landmark_list.size();++j){
			LandmarkObs candidate_landmark;
			candidate_landmark.x = map_landmarks.landmark_list[j].x_f;
			candidate_landmark.y = map_landmarks.landmark_list[j].y_f;
			candidate_landmark.id = map_landmarks.landmark_list[j].id_i;

			// If landmark within particle range
			if ( sqrt(fabs(p_x-candidate_landmark.x)*fabs(p_x-candidate_landmark.x)+fabs(p_y-candidate_landmark.y)*fabs(p_y-candidate_landmark.y)) <= sensor_range ){
			//if (fabs(candidate_landmark.x - p_x) <= sensor_range && fabs(candidate_landmark.y - p_y) <= sensor_range) {
			    // Add candidate landmark to vector
				predLandmarkInRange.push_back(candidate_landmark);
			}

		}

		// Transform observation from vehicle coordinates to map coordinates
		vector<LandmarkObs> transformed_observations;
		for (int j=0;j<observations.size();++j){
			LandmarkObs t_observation;
			t_observation.id = observations[j].id;
			t_observation.x = p_x + observations[j].x*cos(p_theta) - observations[j].y*sin(p_theta);
			t_observation.y = p_y + observations[j].x*sin(p_theta) + observations[j].y*cos(p_theta);
			// Add transformed to map coordinates landmark to vector
			transformed_observations.push_back(t_observation);
		}

		// Associate predictions and observations
		dataAssociation(predLandmarkInRange, transformed_observations);

		// Initialize particle weight
		particles[i].weight = 1.0f;

		for (int j=0;j<transformed_observations.size();++j){
			double t_o_x = transformed_observations[j].x;
			double t_o_y = transformed_observations[j].y;
			double l_x,l_y;
			int associated_landmark_id = transformed_observations[j].id;

			// Find coordinates of associative predicted landmark
			for (int l=0;l<predLandmarkInRange.size();++l){
				if (associated_landmark_id == predLandmarkInRange[l].id){
					l_x = predLandmarkInRange[l].x;
					l_y = predLandmarkInRange[l].y;
				}
			}

			// Calculate weight for current observation
			double std_x = std_landmark[0];
			double std_y = std_landmark[1];
			double normalizer = 2.0*M_PI*std_x*std_y;
			double dx = t_o_x - l_x;
			double dy = t_o_y - l_y;
			double observation_weight = exp( -( (dx*dx)/(2*std_x*std_x) + (dy*dy)/(2*std_y*std_y) ) )/normalizer;

			// Multiply all the calculated measurement probabilities together
			particles[i].weight *= observation_weight;

		}
		// Add particle weight to weight vector
		weights[i] = particles[i].weight;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::discrete_distribution<int> d_d(weights.begin(), weights.end() );
	std::vector<Particle> resampled_particles;

	for (int i=0;i<num_particles;i++){
		//int ind = d_d(gen);
		//Particle p = particles[d_d(gen)];
		resampled_particles.push_back(particles[d_d(gen)]);
	}

	particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
