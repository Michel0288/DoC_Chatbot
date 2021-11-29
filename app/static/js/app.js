jQuery(document).ready(function() {
  console.log("Hello")
    $("#SendButton").click(function(e) {
      e.preventDefault();


      $.ajax({
          type: "POST",
          url: "/chatbot",
          data: {
              question: $("#UserInput").val()
           
          },
          success: function(result) {
            $("#BotResponse").append("<br><br>Me: "+$("#UserInput").val()+ "<br><br> CompBot: "+result.response)+"<br><br><br>";
            $("#UserInput").val("")
          },
          error: function(result) {
              alert('error');
          }
      });



    });

  });