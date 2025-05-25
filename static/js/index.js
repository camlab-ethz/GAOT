window.HELP_IMPROVE_VIDEOJS = false;


$(document).ready(function() {
    // Check for click events on the navbar burger icon

    var options = {
			slidesToScroll: 1,
			slidesToShow: 1,
			loop: true,
			infinite: true,
			autoplay: true,
			autoplaySpeed: 5000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
	
    bulmaSlider.attach();

})

// In your main JS file (e.g., static/js/index.js)
// Make sure this runs after the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Previous carousel init
    if (document.getElementById('accuracy-carousel')) {
        bulmaCarousel.attach('#accuracy-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            pagination: true,
            navigation: true, // Enables prev/next arrows
        });
    }

    // New visualization carousel init
    if (document.getElementById('visualization-carousel')) {
        bulmaCarousel.attach('#visualization-carousel', {
            slidesToScroll: 1,
            slidesToShow: 1,
            loop: true,
            pagination: true,
            navigation: true, // Enables prev/next arrows
        });
    }
});