var loc = '35.731252, 139.730291' // Tokyo expressed as lat,lng tuple
var targetDate = new Date() // Current date/time of user computer
var timestamp = targetDate.getTime()/1000 + targetDate.getTimezoneOffset() * 60 // Current UTC date/time expressed as seconds since midnight, January 1, 1970 UTC
var apikey = 'YOUR_TIMEZONE_API_KEY_HERE'
 
var apicall = 'https://maps.googleapis.com/maps/api/timezone/json?location=' + loc + '&timestamp=' + timestamp + '&key=' + apikey
