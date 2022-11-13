const test = (id) => alert(id);

const get_params = () => {
	axios('http://127.0.0.1:8000/get_params')
		.then(({ data }) => {
			console.log(data);
			$('#speed').text(`${data.speed} m/s`);
			$('#height').text(`${data.height} m`);
			$('#battery').text(`${data.battery} %`);
			$('#temperature').text(`${data.temperature} °F`);
			$('#yaw').text(`${data.yaw} °`);
			$('#flight_time').text(`${data.flight_time} min`);
		})
		.catch((error) => console.log(error));
};

setInterval(function () {
	get_params();
}, 5000);

/*
	Actions:
		· Take off
		· Land
		· Move:
			- Right
			- Left
			- Forward
			- Backwards
			- Up
			- Down
		· Take photo
*/
const perform_action = (action) => {
	console.log(action.toUpperCase());
	axios(`http://127.0.0.1:8000/${action}`)
		.then(({ data }) => console.log(data))
		.catch((error) => console.log(error));
};
