import React from 'react';
import { Link } from 'react-router-dom';
import { Plane, Github, Heart } from 'lucide-react';

export default function Footer() {
  return (
    <footer className="border-t border-border/50 bg-card/50 backdrop-blur-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Brand */}
          <div>
            <div className="flex items-center gap-2.5 mb-4">
              <div className="w-8 h-8 rounded-xl bg-primary flex items-center justify-center">
                <Plane className="w-4 h-4 text-primary-foreground -rotate-45" />
              </div>
              <span className="font-heading font-bold text-lg">FlyPredict</span>
            </div>
            <p className="text-sm text-muted-foreground leading-relaxed max-w-xs">
              ML-powered flight price predictions using Random Forest with 96.8% R2 accuracy.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h4 className="font-heading font-semibold text-sm mb-4 uppercase tracking-wider text-muted-foreground">
              Quick Links
            </h4>
            <div className="space-y-2.5">
              {[
                { label: 'Home', path: '/' },
                { label: 'Predict Price', path: '/predict' },
                { label: 'Model Dashboard', path: '/dashboard' },
                { label: 'About', path: '/about' },
              ].map((link) => (
                <Link
                  key={link.path}
                  to={link.path}
                  className="block text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                  {link.label}
                </Link>
              ))}
            </div>
          </div>

          {/* Project */}
          <div>
            <h4 className="font-heading font-semibold text-sm mb-4 uppercase tracking-wider text-muted-foreground">
              Project
            </h4>
            <div className="space-y-2.5">
              <a
                href="https://github.com/ReddyNikhilG/FLIGHT-TICKET-PRICE-PREDICTION"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors"
              >
                <Github className="w-4 h-4" />
                View on GitHub
              </a>
              <p className="text-sm text-muted-foreground">
                Built with Python, Flask, Scikit-learn & React
              </p>
            </div>
          </div>
        </div>

        <div className="mt-10 pt-6 border-t border-border/50 flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-muted-foreground">
            © 2026 FlyPredict. Educational project by Gali Reddy Nikhil.
          </p>
          <p className="text-xs text-muted-foreground flex items-center gap-1">
            Made with <Heart className="w-3 h-3 text-destructive fill-destructive" /> for learning
          </p>
        </div>
      </div>
    </footer>
  );
}
