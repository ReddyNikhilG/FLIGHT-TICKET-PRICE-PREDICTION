import React from 'react';
import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { motion } from 'framer-motion';
import { Github, GraduationCap, Brain, Code, Plane, ExternalLink } from 'lucide-react';

export default function About() {
  return (
    <div className="min-h-screen py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="font-heading text-3xl sm:text-4xl font-bold tracking-tight">
            About This Project
          </h1>
          <p className="mt-3 text-muted-foreground text-lg max-w-2xl mx-auto">
            A Machine Learning-based web application for predicting Indian domestic flight ticket prices.
          </p>
        </div>

        <div className="space-y-6">
          {/* Author Card */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
            <Card className="border-border/50 overflow-hidden">
              <CardContent className="p-8">
                <div className="flex flex-col sm:flex-row items-center sm:items-start gap-6">
                  <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary to-accent flex items-center justify-center text-white text-2xl font-heading font-bold shrink-0">
                    GN
                  </div>
                  <div className="text-center sm:text-left">
                    <h2 className="font-heading text-2xl font-bold">Gali Reddy Nikhil</h2>
                    <p className="text-muted-foreground mt-1">B.Tech Computer Science (AI & ML)</p>
                    <div className="mt-4 flex flex-wrap gap-3 justify-center sm:justify-start">
                      <a
                        href="https://github.com/ReddyNikhilG"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Button variant="outline" size="sm" className="rounded-full gap-2">
                          <Github className="w-4 h-4" />
                          GitHub Profile
                        </Button>
                      </a>
                      <a
                        href="https://github.com/ReddyNikhilG/FLIGHT-TICKET-PRICE-PREDICTION"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        <Button variant="outline" size="sm" className="rounded-full gap-2">
                          <ExternalLink className="w-4 h-4" />
                          View Repository
                        </Button>
                      </a>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          {/* Project Overview */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }}>
            <Card className="border-border/50">
              <CardContent className="p-8">
                <h3 className="font-heading text-xl font-bold flex items-center gap-2 mb-4">
                  <Plane className="w-5 h-5 text-primary -rotate-45" />
                  Project Overview
                </h3>
                <p className="text-muted-foreground leading-relaxed">
                  Flight ticket prices fluctuate based on multiple factors such as demand, time, 
                  airline, and route. This project aims to build a predictive model that helps users 
                  estimate ticket prices in advance and make better travel decisions. The system uses 
                  historical flight data (22,000+ records) and applies machine learning algorithms to 
                  predict the fare accurately.
                </p>
              </CardContent>
            </Card>
          </motion.div>

          {/* Tech Details Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
              <Card className="border-border/50 h-full">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mx-auto mb-4">
                    <Brain className="w-6 h-6 text-primary" />
                  </div>
                  <h4 className="font-heading font-semibold mb-2">ML Model</h4>
                  <p className="text-sm text-muted-foreground">
                    Random Forest Regressor with 96.8% R² score, trained using Scikit-learn with cross-validation.
                  </p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
              <Card className="border-border/50 h-full">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 rounded-xl bg-accent/10 flex items-center justify-center mx-auto mb-4">
                    <Code className="w-6 h-6 text-accent" />
                  </div>
                  <h4 className="font-heading font-semibold mb-2">Backend</h4>
                  <p className="text-sm text-muted-foreground">
                    Flask API serving predictions with CORS support. Deployed on Render with automatic model loading.
                  </p>
                </CardContent>
              </Card>
            </motion.div>

            <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
              <Card className="border-border/50 h-full">
                <CardContent className="p-6 text-center">
                  <div className="w-12 h-12 rounded-xl bg-chart-3/10 flex items-center justify-center mx-auto mb-4">
                    <GraduationCap className="w-6 h-6 text-chart-3" />
                  </div>
                  <h4 className="font-heading font-semibold mb-2">Educational</h4>
                  <p className="text-sm text-muted-foreground">
                    Created for learning purposes covering the full ML pipeline from data collection to deployment.
                  </p>
                </CardContent>
              </Card>
            </motion.div>
          </div>

          {/* Dataset Features */}
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
            <Card className="border-border/50">
              <CardContent className="p-8">
                <h3 className="font-heading text-xl font-bold mb-4">Dataset Features</h3>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    'Airline', 'Source City', 'Destination City', 'Departure Time',
                    'Arrival Time', 'Stops', 'Class', 'Duration',
                    'Days Left', 'Price (Target)'
                  ].map((feature) => (
                    <div
                      key={feature}
                      className="px-3 py-2 rounded-lg bg-muted/50 border border-border/50 text-sm text-center font-medium"
                    >
                      {feature}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
}
