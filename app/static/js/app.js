$(document).ready(function() {
  $("#Send").click(function(e) {
    e.preventDefault();
  // console.log("Hello")

  $.ajax("/chatbot", {
      type: "POST",
      data: {
        userinput: $("#userinput").val()
      }
    }).done(function(result) {
        // $("#weather-temp").html("<strong>" + result +"</strong> degrees");
        // $("#userinput").val="";
        var sender_reply ='<div class="clear"></div> <div class="msg-list sender"> <div class="messenger-container"> <p>'+$("#userinput").val()+'</p></div></div>';

        var bot_reply ='<div class="clear"></div><div class="msg-list"><div class="messenger-container"><p>'+result.response+'</p></div></div>';

        $("#chatarea").append(sender_reply + bot_reply);
        $("#userinput").val("")
    }).fail(function(result) {
        // $("#message").html("There seems to be an error.");
        alert('error');
    });


  });

  });